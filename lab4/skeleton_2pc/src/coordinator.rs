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
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use coordinator::ipc_channel::ipc::IpcSender;
use coordinator::ipc_channel::ipc::IpcReceiver;
use coordinator::ipc_channel::ipc::TryRecvError;

use message::MessageType;
use message::ProtocolMessage;
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
        self.participants.insert(id, (tx.clone(), rx));
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
        // Add the client to the coordinator's map
        self.clients.insert(id, (tx, rx));
    }
    

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        let successful_ops: u64 = self.successful_ops;
        let failed_ops: u64 = self.failed_ops;
        let unknown_ops: u64 = self.unknown_ops;
        println!("coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", successful_ops, failed_ops, unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        let mut state = CoordinatorState::Quiescent;
        let mut waited_time = 0;
        let mut current_msg: Option<ProtocolMessage> = None;
        let mut current_id: u32 = 0;
        let mut participant_votes: HashMap<u32, bool> = HashMap::new();

        while self.running.load(Ordering::SeqCst) {
            match state {
                CoordinatorState::Quiescent => {
                    participant_votes.clear();
                    for (client_id, (_, rx)) in self.clients.iter() {
                        match rx.try_recv() {
                            Ok(msg) => {
                                //println!("Received client request from client {}: {:?}", client_id, msg);
                                current_msg = Some(msg);
                                current_id = *client_id;
                                state = CoordinatorState::ReceivedRequest;
                                break;
                            }
                            Err(TryRecvError::Empty) => continue,
                            Err(e) => info!("Error receiving from client {}: {:?}", client_id, e),
                        }
                    }
                }
                CoordinatorState::ReceivedRequest => {
                    if let Some(ref msg) = current_msg {
                        self.log.append(
                            MessageType::CoordinatorPropose,
                            msg.txid.clone(),
                            "coordinator".to_string(),
                            0
                        );
                        
                        let txid = msg.txid.clone();
                        let prepare_msg = ProtocolMessage::generate(
                            MessageType::CoordinatorPropose,
                            txid,
                            "coordinator".to_string(),
                            0,
                        );
                        for (participant_id, (tx, _)) in &self.participants {
                            match tx.send(prepare_msg.clone()) {
                                Ok(_) => info!("Sent PREPARE to participant {}", participant_id),
                                Err(e) => info!("Error sending prepare to participant {}: {:?}", participant_id, e),
                            }
                        }
                        state = CoordinatorState::ProposalSent;
                        waited_time = 0;
                    }
                }
                CoordinatorState::ProposalSent => {
                    // Check for votes from all participants
                    for (participant_id, (_, rx)) in &self.participants {
                        match rx.try_recv() {
                            Ok(vote) => {
                                match vote.mtype {
                                    MessageType::ParticipantVoteCommit => {
                                        if current_msg.as_ref().unwrap().txid != vote.txid {
                                            warn!("vote transaction_txid {} != current.txid {}", 
                                                vote.txid, current_msg.as_ref().unwrap().txid);
                                        } else {
                                            participant_votes.insert(*participant_id, true);
                                        }
                                        
                                    }
                                    MessageType::ParticipantVoteAbort => {
                                        state = CoordinatorState::ReceivedVotesAbort;
                                        break;
                                    }
                                    _ => {} 
                                }
                            }
                            Err(TryRecvError::Empty) => continue,
                            Err(e) => {
                                info!("DEBUG: Error receiving vote from participant {} for transaction {}: {:?}", 
                                    participant_id, current_msg.as_ref().unwrap().txid, e);
                                state = CoordinatorState::ReceivedVotesAbort;
                                break;
                            }
                        }
                    }

                    // Check if we have all votes and they're all commits
                    if participant_votes.len() == self.participants.len() 
                        && participant_votes.values().all(|&v| v) {
                        if state != CoordinatorState::ReceivedVotesAbort {
                            state = CoordinatorState::ReceivedVotesCommit;
                        }
                    }

                    // Handle timeout
                    waited_time += 1;
                    if waited_time > 5 {
                        if let Some(ref msg) = current_msg {
                            self.log.append(
                                MessageType::CoordinatorAbort,
                                msg.txid.clone(),
                                "coordinator".to_string(),
                                0
                            );
                        }
                        state = CoordinatorState::ReceivedVotesAbort;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                CoordinatorState::ReceivedVotesCommit => {
                    if let Some(ref msg) = current_msg {
                        println!("DEBUG: Committing transaction {} with votes: {:?}", 
                            msg.txid, participant_votes);
                        self.broadcast_global_decision(
                            msg.txid.clone(),
                            current_id,
                            MessageType::CoordinatorCommit,
                        );
                        self.log.append(
                            MessageType::CoordinatorCommit,
                            msg.txid.clone(),
                            "coordinator".to_string(),
                            0
                        );
                        state = CoordinatorState::SentGlobalDecision;
                        self.successful_ops += 1;
                    }
                }
                CoordinatorState::ReceivedVotesAbort => {
                    if let Some(ref msg) = current_msg {
                        self.broadcast_global_decision(
                            msg.txid.clone(),
                            current_id,
                            MessageType::CoordinatorAbort,
                        );
                        self.log.append(
                            MessageType::CoordinatorAbort,
                            msg.txid.clone(),
                            "coordinator".to_string(),
                            0
                        );
                        state = CoordinatorState::SentGlobalDecision;
                        self.failed_ops += 1;
                    }
                }
                CoordinatorState::SentGlobalDecision => {
                    // Reset to Quiescent for the next transaction
                    current_msg = None;
                    current_id = 0;
                    state = CoordinatorState::Quiescent;
                }
            }
        }
        self.report_status();
    }

    /// Send the global decision (commit or abort) to all participants and clients.
    fn broadcast_global_decision(&mut self, txid: String, client_id: u32, decision: MessageType) {
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

        // Send only to the client that initiated the transaction
        if let Some((tx, _)) = self.clients.get(&client_id) {
            if let Err(e) = tx.send(client_message) {
                error!("Failed to send result to client {}: {:?}", client_id, e);
            }
        } else {
            error!("Client {} no longer exists", client_id);
        }
        info!("Broadcasted global decision {:?} for transaction {}", decision, txid);
    }
    

    pub fn start(&mut self) {
        self.protocol();
    }
}