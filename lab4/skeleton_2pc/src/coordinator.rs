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

use coordinator::ipc_channel::ipc::IpcSender;
use coordinator::ipc_channel::ipc::IpcReceiver;
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
    participants: HashMap<u32, (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>)>,
    clients: HashMap<u32, (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>)>,
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
        r: &Arc<AtomicBool>,
    ) -> Coordinator {
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
    pub fn participant_join(&mut self, id: u32, tx: IpcSender<ProtocolMessage>, rx: IpcReceiver<ProtocolMessage>) {
        println!("\n=== Starting participant_join for ID: {} ===", id);
        
        // Send handshake to the new participant
        let handshake = ProtocolMessage::generate(
            MessageType::HandShake,
            "0".to_string(),
            "coordinator".to_string(),
            0
        );
        println!("Generated handshake message: {:?}", handshake);

        if let Err(e) = tx.send(handshake) {
            error!("Failed to send handshake to participant {}: {:?}", id, e);
        } else {
            println!("Handshake sent successfully to client/participant {}", id);
        }
        
        self.participants.insert(id, (tx.clone(), rx));
        println!("Participant {} joined successfully", id);
    }

    ///
    /// client_join()
    /// Adds a new client for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn client_join(&mut self, id: u32, tx: IpcSender<ProtocolMessage>, rx: IpcReceiver<ProtocolMessage>) {
        if self.state != CoordinatorState::Quiescent {
            error!(
                "Coordinator is not in a quiescent state, cannot accept client {}. Current state: {:?}",
                id, self.state
            );
            return;
        }
    
        // Create and send a handshake message
        let handshake = ProtocolMessage::generate(
            MessageType::HandShake,
            "0".to_string(),  // Transaction ID for handshake
            "coordinator".to_string(),  // Sender ID
            0,  // Operation ID
        );
        println!("Generated handshake message: {:?}", handshake);
        
        if let Err(e) = tx.send(handshake.clone()) {
            error!("Failed to send handshake: {:?}", e);
        } else {
            info!("Handshake sent to client/participant {}", id);
        }
        
    
        match tx.send(handshake.clone()) {
            Ok(_) => println!("Handshake sent to client {}", id),
            Err(e) => {
                error!("Failed to send handshake to client {}: {:?}", id, e);
                return; // Do not add the client if handshake fails
            }
        }
    
        // Add the client to the coordinator's map
        self.clients.insert(id, (tx, rx));
        info!("Client {} successfully joined", id);
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
        println!("Starting coordinator protocol");

        println!(
            "Starting coordinator protocol. Registered clients: {}, participants: {}",
            self.clients.len(),
            self.participants.len()
        );
        
        let transactions_completed = 0;
        let max_transactions = 10;

        while self.running.load(Ordering::SeqCst) && transactions_completed < max_transactions {
            println!("\n=== Transaction Loop Iteration {} ===", transactions_completed + 1);
            
            // Collect messages from clients
            let mut client_messages = Vec::new();
            for (client_id, (_, rx)) in &self.clients {
                match rx.try_recv() {
                    Ok(msg) => {
                        println!("Received message from client {}: {:?}", client_id, msg);
                        client_messages.push(msg);
                    }
                    Err(TryRecvError::Empty) => continue,
                    Err(e) => {
                        println!("Error receiving from client {}: {:?}", client_id, e);
                    }
                }
            }

            // Collect messages from participants
            let mut participant_messages = Vec::new();
            for (participant_id, (_, rx)) in &self.participants {
                match rx.try_recv() {
                    Ok(msg) => {
                        println!("Received message from participant {}: {:?}", participant_id, msg);
                        participant_messages.push(msg);
                    }
                    Err(TryRecvError::Empty) => continue,
                    Err(e) => {
                        println!("Error receiving from participant {}: {:?}", participant_id, e);
                    }
                }
            }

            // Handle collected messages
            for msg in client_messages {
                self.handle_message(msg);
            }
            for msg in participant_messages {
                self.handle_message(msg);
            }

            thread::sleep(Duration::from_millis(100));
        }
    }

    // Add a helper method to handle messages
    fn handle_message(&mut self, msg: ProtocolMessage) {
        match msg.mtype {
            MessageType::HandShake => {
                println!("👍 Received Handshake from client {}", msg.senderid);
                return;
            }
            MessageType::ClientRequest => {
                println!("\n=== Processing ClientRequest ===");
                println!("📥 Received client request: {:?}", msg);
                println!("🔍 Request details - TxID: {}, Sender: {}, OpID: {}", 
                       msg.txid, msg.senderid, msg.opid);
                println!("📊 Current State: {:?}", self.state);
                
                self.state = CoordinatorState::ReceivedRequest;
                println!("⚡ State changed to: {:?}", self.state);
                
                // Log the transaction start
                self.log.append(MessageType::ClientRequest, msg.txid.clone(), "START".to_string(), 0);
                println!("📝 Logged transaction start");
                
                // Create PREPARE message
                let prepare_msg = ProtocolMessage::generate(
                    MessageType::CoordinatorPropose,
                    msg.txid.clone(),
                    "coordinator".to_string(),
                    0
                );
                println!("📬 Created PREPARE message: {:?}", prepare_msg);
                
                // Send PREPARE to all participants only
                println!("\n🔄 Starting to send PREPARE to participants...");
                println!("👥 Number of participants: {}", self.participants.len());
                
                let mut prepare_failed = false;
                for (participant_id, (tx, _)) in &self.participants {
                    println!("📤 Attempting to send PREPARE to participant {}", participant_id);
                    match tx.send(prepare_msg.clone()) {
                        Ok(_) => println!("✅ Successfully sent PREPARE to participant {}", participant_id),
                        Err(e) => {
                            println!("❌ Failed to send PREPARE to participant {}: {:?}", participant_id, e);
                            prepare_failed = true;
                            break;
                        }
                    }
                }
                
                if prepare_failed {
                    println!("\n⚠️ Prepare phase failed, initiating abort...");
                    // If prepare failed, abort immediately
                    let client_abort_msg = ProtocolMessage::generate(
                        MessageType::ClientResultAbort,
                        msg.txid.clone(),
                        "coordinator".to_string(),
                        0
                    );
                    println!("📝 Created abort message: {:?}", client_abort_msg);
                    
                    // Send abort to the requesting client
                    println!("🔍 Looking for client with ID: {}", msg.opid);
                    if let Some((client_tx, _)) = self.clients.get(&msg.opid) {
                        println!("📤 Attempting to send abort to client {}", msg.opid);
                        if let Err(e) = client_tx.send(client_abort_msg) {
                            println!("❌ Failed to send abort to client {}: {:?}", msg.opid, e);
                        } else {
                            println!("✅ Successfully sent abort to client {}", msg.opid);
                        }
                    } else {
                        println!("❌ Could not find client with ID: {}", msg.opid);
                    }
                    
                    self.failed_ops += 1;
                    println!("📊 Updated failed_ops count: {}", self.failed_ops);
                    
                    self.state = CoordinatorState::SentGlobalDecision;
                    println!("⚡ State changed to: {:?}", self.state);
                } else {
                    self.state = CoordinatorState::ProposalSent;
                    println!("\n✅ All PREPARE messages sent successfully");
                    println!("⚡ State changed to: {:?}", self.state);
                    println!("✈️ Transaction {} is now in proposal phase", msg.txid);
                }
                println!("=== ClientRequest Processing Complete ===\n");
            },
            MessageType::ParticipantVoteCommit => {
                println!("👍 Received VOTE_COMMIT from participant {}", msg.senderid);
                
                // Only process votes if we're in the right state
                if self.state != CoordinatorState::ProposalSent {
                    println!("⚠️ Received vote in incorrect state: {:?}", self.state);
                    return;
                }
                
                // Log the vote
                self.log.append(MessageType::ClientRequest, msg.txid.clone(), format!("VOTE_COMMIT from {}", msg.senderid), 0);
                
                // Check if all participants have voted commit
                let all_committed = self.participants.len() > 0 && 
                    self.participants.iter().all(|(_, (_, rx))| {
                        match rx.try_recv() {
                            Ok(vote) => vote.mtype == MessageType::ParticipantVoteCommit,
                            _ => false
                        }
                    });
                
                if all_committed {
                    // All participants voted to commit
                    println!("✅ All participants voted to commit for transaction {}", msg.txid);
                    
                    // Log the decision
                    self.log.append(MessageType::CoordinatorCommit, msg.txid.clone(), "GLOBAL_COMMIT".to_string(), 0);
                    
                    // Send global commit to all participants
                    self.broadcast_global_decision(msg.txid, MessageType::CoordinatorCommit);
                    
                    // Update coordinator state and stats
                    self.state = CoordinatorState::SentGlobalDecision;
                    self.successful_ops += 1;
                }
            },
            MessageType::ParticipantVoteAbort => {
                println!("👎 Received VOTE_ABORT from participant {}", msg.senderid);
                
                // Only process votes if we're in the right state
                if self.state != CoordinatorState::ProposalSent {
                    println!("⚠️ Received vote abort in incorrect state: {:?}", self.state);
                    return;
                }
                
                // Log the vote abort
                self.log.append(MessageType::ParticipantVoteAbort, msg.txid.clone(), format!("VOTE_ABORT from {}", msg.senderid), 0);
                
                // Since we received an abort vote, we can immediately abort the transaction
                println!("❌ Participant {} voted to abort - aborting transaction {}", msg.senderid, msg.txid);
                
                // Log the global abort decision
                self.log.append(MessageType::CoordinatorAbort, msg.txid.clone(), "GLOBAL_ABORT".to_string(), 0);
                
                // Send global abort to all participants
                let txid = msg.txid.clone();
                self.broadcast_global_decision(txid, MessageType::CoordinatorAbort);
                
                // Update coordinator state and stats
                self.state = CoordinatorState::ReceivedVotesAbort;
                self.failed_ops += 1;
                
                println!("❌ Transaction {} aborted due to participant {} vote", msg.txid, msg.senderid);
            },
            MessageType::CoordinatorCommit => {
                println!("✅ Processing COORDINATOR_COMMIT for transaction {}", msg.txid);
                
                // Log the commit decision
                self.log.append(MessageType::CoordinatorCommit, msg.txid.clone(), "GLOBAL_COMMIT".to_string(), 0);
                
                // Create commit messages - different types for participants and clients
                let participant_msg = ProtocolMessage::generate(
                    MessageType::CoordinatorCommit,
                    msg.txid.clone(),
                    "coordinator".to_string(),
                    0
                );
                
                let client_msg = ProtocolMessage::generate(
                    MessageType::ClientResultCommit,  // Changed for clients
                    msg.txid.clone(),
                    "coordinator".to_string(),
                    0
                );
                
                // Notify all participants
                for (participant_id, (tx, _)) in &self.participants {
                    if let Err(e) = tx.send(participant_msg.clone()) {
                        println!("⚠️ Failed to send commit to participant {}: {:?}", participant_id, e);
                    }
                }
                
                // Notify all clients with ClientResultCommit
                for (client_id, (tx, _)) in &self.clients {
                    if let Err(e) = tx.send(client_msg.clone()) {
                        println!("⚠️ Failed to send commit to client {}: {:?}", client_id, e);
                    }
                }
                
                // Update coordinator state and stats
                self.successful_ops += 1;
                self.state = CoordinatorState::SentGlobalDecision;
                
                println!("✅ Commit complete for transaction {}", msg.txid);
            },
            MessageType::CoordinatorAbort => {
                println!("❌ Processing COORDINATOR_ABORT for transaction {}", msg.txid);
                
                // Log the abort decision
                self.log.append(MessageType::CoordinatorAbort, msg.txid.clone(), "GLOBAL_ABORT".to_string(), 0);
                
                // Create abort messages - different types for participants and clients
                let participant_msg = ProtocolMessage::generate(
                    MessageType::CoordinatorAbort,
                    msg.txid.clone(),
                    "coordinator".to_string(),
                    0
                );
                
                let client_msg = ProtocolMessage::generate(
                    MessageType::ClientResultAbort,  // Changed for clients
                    msg.txid.clone(),
                    "coordinator".to_string(),
                    0
                );
                
                // Notify all participants
                for (participant_id, (tx, _)) in &self.participants {
                    if let Err(e) = tx.send(participant_msg.clone()) {
                        println!("⚠️ Failed to send abort to participant {}: {:?}", participant_id, e);
                    }
                }
                
                // Notify all clients with ClientResultAbort
                for (client_id, (tx, _)) in &self.clients {
                    if let Err(e) = tx.send(client_msg.clone()) {
                        println!("⚠️ Failed to send abort to client {}: {:?}", client_id, e);
                    }
                }
                
                // Update coordinator state and stats
                self.failed_ops += 1;
                self.state = CoordinatorState::SentGlobalDecision;
                
                println!("❌ Abort complete for transaction {}", msg.txid);
            },
            _ => {
                println!("⚠️ Unhandled message type: {:?}", msg.mtype);
            }
        }
    }

    /// Send the global decision (commit or abort) to all participants and clients.
    fn broadcast_global_decision(&mut self, txid: String, decision: MessageType) {
        // Create different messages for participants and clients
        let participant_message = ProtocolMessage::generate(
            decision.clone(),  // CoordinatorCommit or CoordinatorAbort
            txid.clone(),
            "coordinator".to_string(),
            0
        );

        // Convert coordinator decision to client result type
        let client_decision = match decision {
            MessageType::CoordinatorCommit => MessageType::ClientResultCommit,
            MessageType::CoordinatorAbort => MessageType::ClientResultAbort,
            _ => panic!("Invalid decision type for broadcast")
        };

        let client_message = ProtocolMessage::generate(
            client_decision,
            txid.clone(),
            "coordinator".to_string(),
            0
        );
    
        // Send to participants
        for (participant_id, (tx, _)) in &self.participants {
            if let Err(e) = tx.send(participant_message.clone()) {
                error!("Failed to send global decision to participant {}: {:?}", participant_id, e);
            }
        }

        // Send to clients
        for (client_id, (tx, _)) in &self.clients {
            if let Err(e) = tx.send(client_message.clone()) {
                error!("Failed to send result to client {}: {:?}", client_id, e);
            }
        }
    
        info!("Broadcasted global decision {:?} for transaction {}", decision, txid);
    }
    

    pub fn start(&mut self) {
        // Start the coordinator protocol
        //self.broadcast_global_decision("0".to_string(), MessageType::HandShake);
        self.protocol();
    }
}
