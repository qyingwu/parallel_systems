//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate stderrlog;
extern crate rand;
extern crate ipc_channel;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use std::time::SystemTime;

use coordinator::ipc_channel::ipc::IpcSender as Sender;
use coordinator::ipc_channel::ipc::IpcReceiver as Receiver;
use coordinator::ipc_channel::ipc::TryRecvError;
use coordinator::ipc_channel::ipc::channel;

use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;

/// CoordinatorState
/// States for 2PC state machine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    ReceivedRequest,
    ProposalSent,
    ReceivedVotesAbort,
    ReceivedVotesCommit,
    SentGlobalDecision
}

/// Coordinator
/// Struct maintaining state for coordinator
///maintain a mapping of participants and clients identifiers to their communication channels
#[derive(Debug)]
pub struct Coordinator {
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    participants: HashMap<u32, (Sender<ProtocolMessage>, Receiver<ProtocolMessage>)>,
    clients: HashMap<u32, (Sender<ProtocolMessage>, Receiver<ProtocolMessage>)>,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,
}

///
/// Coordinator
/// Implementation of coordinator functionality
/// Required:
/// 1. new -- Constructor
/// 2. protocol -- Implementation of coordinator side of protocol
/// 3. report_status -- Report of aggregate commit/abort/unknown stats on exit.
/// 4. participant_join -- What to do when a participant joins
/// 5. client_join -- What to do when a client joins
///
impl Coordinator {

    ///
    /// new()
    /// Initialize a new coordinator
    ///
    /// <params>
    ///     log_path: directory for log files --> create a new log there.
    ///     r: atomic bool --> still running?
    ///
    pub fn new(
        log_path: String,
        r: &Arc<AtomicBool>) -> Coordinator {
        // Create the log directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&log_path).parent() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("Failed to create log directory: {}", e));
        }

        let oplog = oplog::OpLog::new(log_path);
        Coordinator {
            state: CoordinatorState::Quiescent,
            log: oplog,
            running: r.clone(),
            // TODO
            participants: HashMap::new(),
            clients: HashMap::new(),
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
        }
    }

    ///
    /// participant_join()
    /// Adds a new participant for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn participant_join(&mut self, id: u32, tx: Sender<ProtocolMessage>, rx: Receiver<ProtocolMessage>) {
        println!("\n=== Starting participant_join for ID: {} ===", id);
        
        assert!(self.state == CoordinatorState::Quiescent);
        println!("State check passed: Coordinator is in Quiescent state");
        
        // Create ping message
        let ping = ProtocolMessage::generate(
            MessageType::CoordinatorPropose,
            "ping".to_string(),
            "coordinator".to_string(),
            0,
        );
        println!("Created ping message: {:?}", ping);
        
        // Send a ping message with better error handling and retries
        println!("Starting connection verification process...");
        let mut connected = false;
        for attempt in 0..3 {
            println!("\nAttempt {} to establish connection:", attempt + 1);
            
            match tx.send(ping.clone()) {
                Ok(_) => {
                    println!("Successfully sent ping to participant {}", id);
                    // Wait for response
                    println!("Waiting for response from participant {}...", id);
                    match rx.try_recv_timeout(Duration::from_secs(5)) {
                        Ok(response) => {
                            println!("Received response from participant {}: {:?}", id, response);
                            match response.mtype {
                                MessageType::ParticipantVoteCommit => {
                                    println!("Participant {} voted to COMMIT", id);
                                    connected = true;
                                    break;
                                }
                                MessageType::ParticipantVoteAbort => {
                                    println!("Participant {} voted to ABORT", id);
                                    connected = true;  // Still consider it connected, just voted abort
                                    break;
                                }
                                _ => {
                                    println!("WARNING: Unexpected message type from participant {}: {:?}", 
                                        id, response.mtype);
                                }
                            }
                        }
                        Err(e) => {
                            println!("ERROR: Failed to receive response from participant {}: {:?}", id, e);
                            warn!("Attempt {} - No response from participant {}: {:?}", 
                                attempt + 1, id, e);
                        }
                    }
                }
                Err(e) => {
                    println!("WARNING: Attempt {} - Failed to ping participant {}: {:?}", 
                        attempt + 1, id, e);
                }
            }
            thread::sleep(Duration::from_millis(100));
        }

        if connected {
            self.participants.insert(id, (tx, rx));
            info!("Participant {} joined successfully", id);
        } else {
            error!("Failed to establish connection with participant {}", id);
        }
    }

    ///
    /// client_join()
    /// Adds a new client for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn client_join(&mut self, id: u32, tx: Sender<ProtocolMessage>, rx: Receiver<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);
        self.clients.insert(id, (tx, rx));
        info!("Client {} joined", id);
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        println!(
            "coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}",
            self.successful_ops, self.failed_ops, self.unknown_ops
        );
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        info!("Starting coordinator protocol");
        let mut transactions_completed = 0;
        let max_transactions = 10; // Or whatever number you want to run
        
        while self.running.load(Ordering::Relaxed) && transactions_completed < max_transactions {
            // Wait for participants to be ready
            if self.participants.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            info!("Starting transaction {}", transactions_completed);
            
            // Create transaction
            let txid = format!("tx_{}", transactions_completed);
            let proposal = ProtocolMessage::generate(
                MessageType::CoordinatorPropose,
                txid.clone(),
                "coordinator".to_string(),
                transactions_completed as u32,
            );

            // Send proposal to all participants
            for (id, (tx, _)) in &self.participants {
                if let Err(e) = tx.send(proposal.clone()) {
                    error!("Failed to send proposal to participant {}: {:?}", id, e);
                    continue;
                }
            }

            // ... rest of the transaction logic ...

            transactions_completed += 1;
            std::thread::sleep(Duration::from_millis(500));
        }

        info!("Coordinator completed {} transactions", transactions_completed);
        self.report_status();
    }

    // Separate shutdown logic into its own function
    pub fn shutdown(&mut self) {
        info!("Coordinator shutting down...");
    
        let shutdown_msg = ProtocolMessage::generate(
            MessageType::CoordinatorExit,
            "shutdown".to_string(),
            "coordinator".to_string(),
            0,
        );
    
        // Notify participants
        for (id, (tx, _)) in self.participants.drain() {
            match tx.send(shutdown_msg.clone()) {
                Ok(_) => info!("Sent shutdown to participant {}", id),
                Err(e) => error!("Failed to notify participant {}: {:?}", id, e),
            }
        }
    
        // Notify clients
        for (id, (tx, _)) in self.clients.drain() {
            match tx.send(shutdown_msg.clone()) {
                Ok(_) => info!("Sent shutdown to client {}", id),
                Err(e) => error!("Failed to notify client {}: {:?}", id, e),
            }
        }
    
        self.report_status();
    }
    

    ///When a client request is received, initiate the 2PC protocol.
    fn handle_client_request(&mut self, client_id: u32, msg: ProtocolMessage) -> Result<(), Box<dyn std::error::Error>> {
        if self.participants.is_empty() {
            return Err("No participants available".to_string().into());
        }

        let txid = &msg.txid;
        info!("Coordinator received request from client {}: {:?}", client_id, msg);
    
        self.state = CoordinatorState::ReceivedRequest;
    
        // Send proposal with timeout
        let proposal = ProtocolMessage::generate(
            MessageType::CoordinatorPropose,
            msg.txid.clone(),
            "coordinator".to_string(),
            0,
        );
    
        let mut active_participants = Vec::new();
        for (id, (tx, _)) in &self.participants {
            match tx.send(proposal.clone()) {
                Ok(_) => active_participants.push(*id),
                Err(e) => {
                    error!("Failed to send proposal to participant {}: {:?}", id, e);
                    return Err(format!("Failed to send proposal: {:?}", e).into());
                }
            }
        }
    
        if active_participants.is_empty() {
            return Err("No active participants".to_string().into());
        }
    
        self.state = CoordinatorState::ProposalSent;
    
        // Collect participant IDs first
        let participant_ids: Vec<u32> = self.participants.keys().cloned().collect();
        
        // Collect votes
        let mut commit_votes = 0;
        let mut abort_votes = 0;

        for participant_id in participant_ids {
            if let Some((_, rx)) = self.participants.get(&participant_id) {
                match rx.try_recv_timeout(Duration::from_secs(5)) {
                    Ok(vote) => match vote.mtype {
                        MessageType::ParticipantVoteCommit => commit_votes += 1,
                        MessageType::ParticipantVoteAbort => abort_votes += 1,
                        _ => warn!("Unexpected vote message type from participant {}", participant_id),
                    },
                    Err(TryRecvError::Empty) => {
                        if !self.running.load(Ordering::Relaxed) {
                            break;
                        }
                        self.remove_participant(&participant_id);
                        abort_votes += 1;
                    },
                    Err(TryRecvError::IpcError(err)) => {
                        error!("Error receiving vote from participant {}: {:?}", participant_id, err);
                        self.remove_participant(&participant_id);
                        abort_votes += 1;
                    }
                }
            }
        }
        
    
        // Decide commit or abort
        let decision = if abort_votes > 0 {
            MessageType::CoordinatorAbort
        } else {
            MessageType::CoordinatorCommit
        };

        self.broadcast_global_decision(msg.txid.clone(), decision);
    
        // Update stats
        match decision {
            MessageType::CoordinatorCommit => self.successful_ops += 1,
            MessageType::CoordinatorAbort => self.failed_ops += 1,
            _ => self.unknown_ops += 1,
        }
    
        self.state = CoordinatorState::SentGlobalDecision;

        // Notify client
        if let Some((client_tx, _)) = self.clients.get(&client_id) {
            let client_result = ProtocolMessage::generate(
                match decision {
                    MessageType::CoordinatorCommit => MessageType::ClientResultCommit,
                    MessageType::CoordinatorAbort => MessageType::ClientResultAbort,
                    _ => unreachable!(),
                },
                msg.txid.clone(),
                "coordinator".to_string(),
                0,
            );

            if let Err(e) = client_tx.send(client_result) {
                error!("Failed to send result to client {}: {:?}", client_id, e);
            }
        }

        Ok(())
    }


    ///Send the global decision (commit or abort) to all participants.
    fn broadcast_global_decision(&mut self, txid: String, decision: MessageType) {
        let decision_message = ProtocolMessage::generate(decision, txid.clone(), "coordinator".to_string(), 0);
    
        for (participant_id, (tx, _)) in &self.participants {
            if let Err(e) = tx.send(decision_message.clone()) {
                error!("Failed to send global decision to participant {}: {:?}", participant_id, e);
            }
        }
    
        info!("Broadcasted global decision {:?} for transaction {}", decision, txid);
    }
    
    
    pub fn setup_channels(&mut self, id: u32, is_participant: bool) -> Result<(Sender<ProtocolMessage>, Receiver<ProtocolMessage>), String> {
        // Add retry logic for channel creation
        let mut retries = 3;
        while retries > 0 {
            match (channel::<ProtocolMessage>(), channel::<ProtocolMessage>()) {
                (Ok((coord_tx, client_rx)), Ok((client_tx, coord_rx))) => {
                    // Store coordinator's channels
                    if is_participant {
                        self.participants.insert(id, (client_tx, coord_rx));
                    } else {
                        self.clients.insert(id, (client_tx, coord_rx));
                    }
                    
                    // Verify channel connection with a ping
                    let ping = ProtocolMessage::generate(
                        MessageType::CoordinatorPropose,
                        "ping".to_string(),
                        "coordinator".to_string(),
                        0,
                    );

                    // Try to send a test message
                    match coord_tx.send(ping) {
                        Ok(_) => {
                            info!("{} {} channels set up successfully", 
                                if is_participant { "Participant" } else { "Client" }, 
                                id);
                            return Ok((coord_tx, client_rx));
                        }
                        Err(_) => {
                            warn!("Channel verification failed, retrying...");
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                }
                _ => {
                    warn!("Failed to create channels, retrying...");
                    thread::sleep(Duration::from_millis(100));
                }
            }
            retries -= 1;
        }
        
        Err("Failed to establish IPC channels after multiple attempts".to_string())
    }

    pub fn start(&mut self) {
        // Start the coordinator protocol
        self.protocol();
    }

    fn remove_participant(&mut self, participant_id: &u32) {
        self.participants.remove(participant_id);
        error!("Participant {} removed due to failure or timeout", participant_id);
    }
    
    

    // Add new method for health checks
    fn check_participant_health(&mut self) {
        let ping = ProtocolMessage::generate(
            MessageType::CoordinatorPropose,
            format!("health_check_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
            "coordinator".to_string(),
            0,
        );

        let participant_ids: Vec<u32> = self.participants.keys().cloned().collect();
        for id in participant_ids {
            if let Some((tx, _)) = self.participants.get(&id) {
                let mut retries = 3;
                while retries > 0 {
                    if tx.send(ping.clone()).is_ok() {
                        debug!("Health check passed for participant {}", id);
                        break;
                    }
                    retries -= 1;
                    thread::sleep(Duration::from_millis(100));
                }
                if retries == 0 {
                    error!("Participant {} failed health check, removing from active participants", id);
                    self.remove_participant(&id);
                }
            }
        }
        
    }

    // Add this new method for better logging
    fn log_status(&self) {
        info!("Coordinator status:");
        info!("Active participants: {}", self.participants.len());
        info!("Active clients: {}", self.clients.len());
        info!("Current state: {:?}", self.state);
    }
}
