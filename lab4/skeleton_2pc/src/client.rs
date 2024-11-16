//!
//! client.rs
//! Implementation of 2PC client
//!
extern crate ipc_channel;
extern crate log;
extern crate stderrlog;

use std::thread;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use client::ipc_channel::ipc::IpcReceiver as Receiver;
use client::ipc_channel::ipc::TryRecvError;
use client::ipc_channel::ipc::IpcSender as Sender;

use message;
use message::MessageType;
use message::RequestStatus;

// Client state and primitives for communicating with the coordinator
#[derive(Debug)]
pub struct Client {
    pub id_str: String,
    pub running: Arc<AtomicBool>,
    pub num_requests: u32,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,
    sender: Sender<message::ProtocolMessage>,
    receiver: Receiver<message::ProtocolMessage>,
    transaction_status: HashMap<String, RequestStatus>,
}

///
/// Client Implementation
/// Required:
/// 1. new -- constructor
/// 2. pub fn report_status -- Reports number of committed/aborted/unknown
/// 3. pub fn protocol(&mut self, n_requests: i32) -- Implements client side protocol
///
impl Client {

    ///
    /// new()
    ///
    /// Constructs and returns a new client, ready to run the 2PC protocol
    /// with the coordinator.
    ///
    /// HINT: You may want to pass some channels or other communication
    ///       objects that enable coordinator->client and client->coordinator
    ///       messaging to this constructor.
    /// HINT: You may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor
    ///
    pub fn new(id_str: String,
               running: Arc<AtomicBool>,
               sender: Sender<message::ProtocolMessage>,
               receiver: Receiver<message::ProtocolMessage>) -> Client {
        Client {
            id_str: id_str,
            running: running,
            num_requests: 0,
            sender,
            receiver,
            failed_ops: 0,
            successful_ops: 0,
            unknown_ops: 0,
            transaction_status: HashMap::new(),
        }
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// Wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("{}::Waiting for exit signal", self.id_str.clone());

        // TODO
        while self.running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
        }

        self.shutdown();
        trace!("{}::Exiting", self.id_str.clone());
    }

    ///
    /// send_next_operation(&mut self)
    /// Send the next operation to the coordinator
    ///
    pub fn send_next_operation(&mut self) {

        // Create a new request with a unique TXID.
        self.num_requests += 1;
        let txid = format!("{}_op_{}", self.id_str.clone(), self.num_requests);
        let pm = message::ProtocolMessage::generate(message::MessageType::ClientRequest,
                                                    txid.clone(),
                                                    self.id_str.clone(),
                                                    self.num_requests);
        info!("{}::Sending operation #{}", self.id_str.clone(), self.num_requests);

        // Send the message using the IPC channel
        if let Err(e) = self.sender.send(pm) {
            error!("{}::Failed to send operation: {}", self.id_str.clone(), e);
            self.transaction_status.insert(txid.clone(), RequestStatus::Unknown);
            self.unknown_ops += 1;
        } else {
            info!("{}::Sent operation #{}", self.id_str.clone(), self.num_requests);
        }
        
    }

    ///
    /// recv_result()
    /// Wait for the coordinator to respond with the result for the
    /// last issued request. Note that we assume the coordinator does
    /// not fail in this simulation
    ///

    pub fn recv_result(&mut self) {

        info!("{}::Receiving Coordinator Result", self.id_str.clone());

        match self.receiver.recv() {
            Ok(msg) => {
                match msg.mtype {
                    MessageType::CoordinatorCommit => {
                        self.transaction_status.insert(msg.txid.clone(), RequestStatus::Committed);
                        self.successful_ops += 1;
                        info!("{}::Received commit for operation: {}", self.id_str.clone(), msg.txid);
                    }
                    MessageType::CoordinatorAbort => {
                        self.transaction_status.insert(msg.txid.clone(), RequestStatus::Aborted);
                        self.failed_ops += 1;
                        info!("{}::Received abort for operation: {}", self.id_str.clone(), msg.txid);
                    }
                    MessageType::CoordinatorExit => {
                        info!("{}::Coordinator is shutting down", self.id_str.clone());
                        self.shutdown();
                    }
                    _ => {
                        self.unknown_ops += 1;
                        error!("{}::Unknown message type received for operation: {}", self.id_str.clone(), msg.txid);
                    }
                }
            },
            Err(e) => {
                error!("{}::Failed to receive response: {:?}", self.id_str.clone(), e);
                self.unknown_ops += 1; // Treat as unknown
            }
        }
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this client before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        println!(
            "{:16}:\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}",
            self.id_str.clone(),
            self.successful_ops,
            self.failed_ops,
            self.unknown_ops
        );
    }

    ///
    /// protocol()
    /// Implements the client side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep issuing requests!
    /// HINT: if you've issued all your requests, wait for some kind of
    ///       exit signal before returning from the protocol method!
    ///
    pub fn protocol(&mut self, n_requests: u32) {
        for _ in 0..n_requests {
            if !self.running.load(Ordering::Relaxed) {
                break;
            }
            self.send_next_operation();

            // Wait for response with timeout
            let mut received_response = false;
            while !received_response && self.running.load(Ordering::Relaxed) {
                match self.receiver.try_recv() {
                    Ok(msg) => {
                        match msg.mtype {
                            MessageType::CoordinatorExit => {
                                info!("{}::CoordinatorExit received, shutting down", self.id_str.clone());
                                self.running.store(false, Ordering::Relaxed);
                                return;
                            }
                            _ => {
                                // Handle normal response
                                self.handle_response(msg);
                                received_response = true;
                            }
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        thread::sleep(Duration::from_millis(100));
                    }
                    Err(e) => {
                        error!("{}::Error receiving from coordinator: {:?}", self.id_str.clone(), e);
                        self.unknown_ops += 1;
                        received_response = true; // Break the loop on error
                    }
                }
            }
        }

        self.wait_for_exit_signal();
        self.report_status();
    }

    // Add this helper method to handle responses
    fn handle_response(&mut self, msg: message::ProtocolMessage) {
        match msg.mtype {
            MessageType::CoordinatorCommit => {
                self.transaction_status.insert(msg.txid.clone(), RequestStatus::Committed);
                self.successful_ops += 1;
                info!("{}::Received commit for operation: {}", self.id_str.clone(), msg.txid);
            }
            MessageType::CoordinatorAbort => {
                self.transaction_status.insert(msg.txid.clone(), RequestStatus::Aborted);
                self.failed_ops += 1;
                info!("{}::Received abort for operation: {}", self.id_str.clone(), msg.txid);
            }
            _ => {
                self.unknown_ops += 1;
                error!("{}::Unknown message type received for operation: {}", self.id_str.clone(), msg.txid);
            }
        }
    }

    pub fn start(&mut self) {
        // Start the client protocol with the configured number of requests
        // You'll need to store n_requests in the Client struct and pass it here
        self.protocol(self.num_requests);
    }

    
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        info!("{}::Shutting down client", self.id_str.clone());
    }

}
