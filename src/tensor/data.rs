use crate::Shape;
use rand::{distributions::Standard, prelude::Distribution};

#[derive(new, Debug, Clone, PartialEq)]
pub struct Data<P, const D: usize> {
    pub value: Vec<P>,
    pub shape: Shape<D>,
}

//this implementation block only works for types where:
//P implements Debug(i.e. can be printed with{:?})
//Standard can generate random P values(like f32, i32, u32,etc.)
impl <P: std::fmt::Debug, const D: usize> Data<P,D>
where
    Standard: Distribution<P>,
{
    pub fn random(shape: Shape<D>) -> Data<P,D> {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(rand::random());
        }

        Data::new(data, shape)
    }
}

impl <P: std::fmt::Debug + Copy, const A: usize> From<[P; A] for Data<P, 1> {
    fn from(elems: [P; A]) -> Self {
        
    }
}