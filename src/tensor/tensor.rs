use std::ops::Range;

use crate::{Data, Shape};

pub enum TensorError {
    ReshapeError(String),
}

pub trait FloatTensor<P: num_traits::Float, const D:usize>:
    TensorBase<P, D>
{

}

pub trait TensorBase<P, const D: usize> {
    //self here is the instance of the type  that implements this trait
    //define a method called shape that takes a reference to self and returns a reference to a Shape with D dimensions
    fn shape(&self) -> &Shape<D>
    fn into_data(self) -> Data<P, D>;
    fn from <O: TensorBase<P, D>>(other: O) -> Self;
    fn empty(shape: Shape<D>) -> Self;
}

//create a public Trait with two generic parameters
pub trait TensorOpsAdd<P, const D: usize>:
    //this trait requires that any implementing type must also implement the standard libary's add trait for self+self operations
    //and the implementing type must also implement the add trait for Self + P operations(P being a scalar getting added to the tensor)
    std::ops::Add<Self, Output = Self> + sd::ops::Add<P, Output = Self>
where
    //with an additional constratint that implementing type must be sized(have a known size at compile time)
{
    fn add(&self, other: &Self)-> Self;
    fn add_scalar(&self, other: &P) -> Self;
}

pub trait TensorOpsMatmul<P, const D: usize> {
    fn matmul(&self, other: &Self) -> Self;
}

pub trait TensorOpsNeg<P, const D: usize>: std::ops::Neg {
    fn neg(&self) -> Self;
}

pub trait TensorOpsMul<P, const D: usize>:
    std::ops::Mul<P, Output=Self> + std::ops::Mul<Self, Output = Self>
where
    Self: Sized,
{
    fn mul(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: &P) -> Self;
}

pub trait TensorOpsReshape<P, const D1: usize, const D2: usize, T: TensorBase<P, D2>> {
    fn reshape(&self, shape: Shape<D2>) -> T;
}

pub trait TensorOpsIndex<P, const D1: usize, const D2: usize> {
    fn index(&self, indexes: [Range<usize>; D2]) -> Self;
}