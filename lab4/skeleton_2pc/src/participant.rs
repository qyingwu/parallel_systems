//!
//! participant.rs
//! Implementation of 2PC participant
//!
extern crate ipc_channel;
extern crate log;
extern crate rand;
extern crate stderrlog;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::thread;

use participant::rand::prelude::*;
use participant::ipc_channel::ipc::IpcReceiver as Receiver;
use participant::ipc_channel::ipc::TryRecvError;
use participant::ipc_channel::ipc::IpcSender as Sender;

use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;

///
/// ParticipantState
/// enum for Participant 2PC state machine
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipantState {
    Quiescent,
    ReceivedP1,
    VotedAbort,
    VotedCommit,
    AwaitingGlobalDecision,
}

///
/// Participant
/// Structure for maintaining per-participant state and communication/synchronization objects to/from coordinator
///
#[derive(Debug)]
pub struct Participant {
    id_str: String,
    state: ParticipantState,
    log: oplog::OpLog,
    running: Arc<AtomicBool>,
    send_success_prob: f64,
    operation_success_prob: f64,
    receiver: Receiver<ProtocolMessage>, // Channel for receiving messages from coordinator
    sender: Sender<ProtocolMessage>,     // Channel for sending messages to coordinator
}

///
/// Participant
/// Implementation of participant for the 2PC protocol
/// Required:
/// 1. new -- Constructor
/// 2. pub fn report_status -- Reports number of committed/aborted/unknown for each participant
/// 3. pub fn protocol() -- Implements participant side protocol for 2PC
///
impl Participant {

    ///
    /// new()
    ///
    /// Return a new participant, ready to run the 2PC protocol with the coordinator.
    ///
    /// HINT: You may want to pass some channels or other communication
    ///       objects that enable coordinator->participant and participant->coordinator
    ///       messaging to this constructor.
    /// HINT: You may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor. There are other
    ///       ways to communicate this, of course.
    ///
    pub fn new(
        id_str: String,
        log_path: String,
        r: Arc<AtomicBool>,
        send_success_prob: f64,
        operation_success_prob: f64,
        receiver: Receiver<ProtocolMessage>,
        sender: Sender<ProtocolMessage>,
    ) -> Participant {

        Participant {
            id_str: id_str,
            state: ParticipantState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r,
            send_success_prob: send_success_prob,
            operation_success_prob: operation_success_prob,
            // TODO
            receiver,
            sender,
        }
    }

    ///
    /// send()
    /// Send a protocol message to the coordinator. This can fail depending on
    /// the success probability. For testing purposes, make sure to not specify
    /// the -S flag so the default value of 1 is used for failproof sending.
    ///
    /// HINT: You will need to implement the actual sending
    ///
    pub fn send(&mut self, pm: ProtocolMessage) {
        let x: f64 = rand::random();
        if x <= self.send_success_prob {
            println!("participant send send_success_prob");
            if let Err(e) = self.sender.send(pm.clone()) {
                error!("{}::Failed to send message to coordinator after retries: {:?}", self.id_str, e);
            }
        } else {
            println!("participant send send_failure");
            warn!("{}::Message failed to send (simulated failure): {:?}", self.id_str, pm);
        }
    }
    
    
    

    ///
    /// perform_operation
    /// Perform the operation specified in the 2PC proposal,
    /// with some probability of success/failure determined by the
    /// command-line option success_probability.
    ///
    /// HINT: The code provided here is not complete--it provides some
    ///       tracing infrastructure and the probability logic.
    ///       Your implementation need not preserve the method signature
    ///       (it's ok to add parameters or return something other than
    ///       bool if it's more convenient for your design).
    ///
    pub fn perform_operation(&mut self, request_option: &Option<ProtocolMessage>) -> bool {
        trace!("{}::Performing operation", self.id_str.clone());
        let x: f64 = random();
        
        // Create success/failure messages
        let message = match request_option {
            Some(req) => ProtocolMessage::generate(
                if x <= self.operation_success_prob { 
                    MessageType::ParticipantVoteCommit 
                } else { 
                    MessageType::ParticipantVoteAbort 
                },
                req.txid.clone(),
                self.id_str.clone(),
                0,
            ),
            None => {
                error!("{}::No request message provided", self.id_str);
                return false;
            }
        };

        if x <= self.operation_success_prob {
            // Operation succeeded
            if let Err(e) = self.sender.send(message) {
                println!("participant operartion send_success");
                error!("{}::Failed to send successful operation message: {:?}", self.id_str, e);
                return false;
            }
            true
        } else {
            // Operation failed
            println!("participant operartion send_failure");
            if let Err(e) = self.sender.send(message) {
                error!("{}::Failed to send failed operation message: {:?}", self.id_str, e);
                return false;
            }
            false
        }
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        let successful_ops = self.log.arc().lock().unwrap()
        .values()
        .filter(|pm| pm.mtype == MessageType::CoordinatorCommit)
        .count() as u64;

        let failed_ops = self.log.arc().lock().unwrap()
            .values()
            .filter(|pm| pm.mtype == MessageType::CoordinatorAbort)
            .count() as u64;

        let unknown_ops = self.log.arc().lock().unwrap()
            .values()
            .filter(|pm| pm.mtype == MessageType::ClientResultAbort || pm.mtype == MessageType::ClientResultCommit)
            .count() as u64;

        println!("{:16}:\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", self.id_str, successful_ops, failed_ops, unknown_ops);
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// Wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("{}::Waiting for exit signal", self.id_str.clone());
        while self.running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
        }
        trace!("{}::Exiting", self.id_str.clone());
    }

    ///
    /// protocol()
    /// Implements the participant side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        trace!("{}::Beginning protocol", self.id_str.clone());
    
        while self.running.load(Ordering::Relaxed) {
            match self.receiver.try_recv() {
                Ok(pm) => {
                    println!("📨 Received message: {:?}", pm.mtype);
                    match pm.mtype {
                        MessageType::CoordinatorPropose => {
                            println!("🔄 Processing CoordinatorPropose request...");
                            self.state = ParticipantState::ReceivedP1;
                            trace!("{}::Received proposal: {:?}", self.id_str, pm);
        
                            // Perform operation and decide vote
                            println!("⚙️ Performing operation...");
                            let vote = if self.perform_operation(&Some(pm.clone())) {
                                println!("✅ Operation successful - voting to commit");
                                MessageType::ParticipantVoteCommit
                            } else {
                                println!("❌ Operation failed - voting to abort");
                                MessageType::ParticipantVoteAbort
                            };
        
                            // Log and send vote
                            println!("📝 Logging vote: {:?}", vote);
                            self.log.append(vote, pm.txid.clone(), self.id_str.clone(), 0);
                            
                            println!("📤 Sending vote to coordinator");
                            self.send(
                                ProtocolMessage::generate(vote, pm.txid.clone(), self.id_str.clone(), 0),
                            );
        
                            self.state = if vote == MessageType::ParticipantVoteCommit {
                                println!("➡️ State changed to VotedCommit");
                                ParticipantState::VotedCommit
                            } else {
                                println!("➡️ State changed to VotedAbort");
                                ParticipantState::VotedAbort
                            };
                        }
                        MessageType::CoordinatorCommit | MessageType::CoordinatorAbort => {
                            println!("📨 Received coordinator decision: {:?}", pm.mtype);
                            trace!("{}::Received global decision: {:?}", self.id_str, pm);
                            println!("📝 Logging coordinator decision");
                            self.log.append(pm.mtype, pm.txid.clone(), self.id_str.clone(), 0);
                            println!("➡️ State changed to Quiescent");
                            self.state = ParticipantState::Quiescent;
                        }
                        MessageType::CoordinatorExit => {
                            println!("🛑 Received exit signal from coordinator");
                            info!("{}::Received CoordinatorExit, shutting down.", self.id_str);
                            self.running.store(false, Ordering::Relaxed);
                            break;
                        }
                        _ => {
                            warn!("{}::Unexpected message type received: {:?}", self.id_str, pm);
                        }
                    }
                },
                Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(100)),
                Err(_) => {
                    error!("{}::Disconnected from coordinator.", self.id_str);
                    break;
                }
            }
        }
    
        self.report_status();
    }

    pub fn start(&mut self) {
        info!("Participant {} starting protocol", self.id_str);
        self.protocol();
    }
}
