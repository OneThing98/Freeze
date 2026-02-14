use std::ops::Range;

#[derive(new, Debug, Clone, PartialEq)]
pub struct Shape<const D: usize> {
    pub dims: [usize; D],
}

impl<const D: usize> Shape<D> {
    pub fn num_elements(&self) -> usize {
        let mut num_elements = 1;
        for i in 0..D {
            num_elements *= self.dims[i];
        }

        num_elements
    }
}

impl<const D1: usize> Shape<D1> {
    pub fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self {
        if D2 > D1 {
            panic!("Cant index that");
        }

        //stire new dimension sizes
        let mut dims = [0; D1];

        //for each axis we slice, we are calculating how many elements will remain after that slice
        for i in 0..D2 {
            dims[i] = indexes[i].clone().count()
        }

        //for the axes we didnt slice(i.e. D2 to D1) keep the original size
        //self.dims[i] is the original tensor's axis sizes
        //dims[i] is the new tensor's axis sizes
        for i in D2..D1 {
            dims[i] = self.dims[i];
        }

        Self::new(dims)
    }
}