use std::{
    cell:RefCell,
    ops::{Add, Mul},
    rc::Rc,
};

//NodeId is the identity system, every node gets a unique id for tracking the graph
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct NodeId {
    value: String,
}

impl NodeId {
    pub fn new() -> Self {
        Self {
            value: nanoid::nanoid!(),
        }
    }
}

//This trait is the contract, it defines what every node in the graph must be able to do
pub trait Node<Out>: std::fmt::Debug {
    fn id(&self) -> NodeId;
    fn grad(&mut self) -> Out;
    fn value(&self) -> Out;
    fn update_grad(&mut self, grad: Out);
}

//uses rust's reference counted smart pointer Rc to enable shared ownershio
//wraps it with interior mutability to allow mutation through immutable references,
//and uses dynamic dispatch to store any concrete type that implements Node trait, creating
//a type that can be shared between multiple graph operations while still allowing gradient updates
//during backpropagation

//this is the sharing mechanism
pub type NodeRef<T> = Rc<RefCell<dyn Node<T>>>;

//any type implementing this trait knows how to create 0 filled version of type T
//these are helper traits
pub trait Zeros<T> {
    fn zeroes(&self) -> T;
}

pub trait Ones<T> {
    fn ones(&self) -> T;
}


//and this is the actual implementation
#[derive(Debug)]
pub struct RootNode<Out> {
    pub id: NodeId,
    pub value: Out,
    pub grad: Option<Out>,
}

impl <Out> RootNode<Out> {
    pub fn new(value: Out) -> Self {
        Self {
            id: NodeId::new()
            value,
            grad: None,
        }
    }
}

impl<Out> Node<Out> for RootNode<Out>
where
    Out: Zeroes<Out> + Clone + Mul<Output = Out> + Add<Output = Out>,
    Out: std::fmt::Debug,
{
    fn id(&self) -> NodeId {
        self.id.clone()
    }

    fn grad(&mut self) -> Out {
        let grad_self: Out = match &self.grad {
            Some(val) => val.clone(),
            None => self.value.zeros(),
        };
        self.grad = Some(grad_self.clone());
        grad_self
    }

    fn value(&self) -> Out {
        self.value.clone()
    }

    fn update_grad(&mut self, grad: Out) {
        self.grad = Some(self.grad() + grad);
    }
}



//so how they would all work together is:

//1. Create Nodes
// let input = RootNode::new(tensor1); //creates a node with unique id
// let weights = RootNode::new(tensor2); //creates another node with different id

//2. Make them shareable
//let input_ref: NodeRef<Tensor> = Rc::new(RefCell:new(input));
//let weights_ref: NodeRef<Tensor> = Rc::new(RefCell::new(weights));

//these can then be used in multiple operations
// Both operations can use the same input
//let op1 = create_add_operation(input_ref.clone(), weights_ref.clone());
//let op2 = create_mul_operation(input_ref.clone(), other_ref);
//                             ^^^^^^^^^^^^^^^^^
//                             Same node used in multiple places!

//or gradient flow
//during backpropagation
//input_ref.borrow_mut().update_grad(gradient_from_op1);  // Add gradient from op1
//input_ref.borrow_mut().update_grad(gradient_from_op2);  // Add gradient from op2
// input now has accumulated gradients from both operations!


#[macro_export]
macro_rules! node_init {
    // pattern 1: Binary operations (takes 2 inputs, produces 1 output)
    // Used for: addition, multiplication, matrix multiplication, etc.
    ( lhs $lhs:expr, rhs $rhs:expr, out $out:expr, ) => {{
        // Import BinaryOpsNode (doesn't exist yet - will be created in future steps)
        use $crate::graph::ops::BinaryOpsNode;
        // Create a binary operation node with left operand, right operand, and output
        let node = BinaryOpsNode::new($lhs, $rhs, $out);
        // Wrap in Rc<RefCell<>> to make it shareable and mutable in the computation graph
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    
    // Pattern 2: Single operations (takes 1 input, produces 1 output)  
    // Used for: negation, reshape, activation functions, etc.
    ( input $input:expr, out $out:expr, ) => {{
        // Import SingleOpsNode (doesn't exist yet - will be created in future steps)
        use $crate::graph::ops::SingleOpsNode;
        // Create a single operation node with input and output
        let node = SingleOpsNode::new($input, $out);
        // Wrap in Rc<RefCell<>> to make it shareable and mutable in the computation graph
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
    
    // Pattern 3: Root nodes (no inputs, just holds data)
    // Used for: input tensors, weight parameters, bias parameters
    ( root $out:expr ) => {{
        // Import RootNode (this exists - we just defined it)
        use $crate::graph::node::RootNode;
        // Create a root node that holds the data/tensor
        let node = RootNode::new($out);
        // Wrap in Rc<RefCell<>> to make it shareable and mutable in the computation graph
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }};
}