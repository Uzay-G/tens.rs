use std::ops::{Div, Mul, Add, Sub, AddAssign};
use crate::ops::*; 
use num_traits::Num;

mod ops;

// we have our tensor type
struct Tensor<'a, T> {
    data: Vec<T>,
    grad: Vec<T>,
    shape: Vec<usize>,
    stride: Vec<usize>,
    op: Option<TensOp>,
    l_parent: Option<&'a Tensor<'a, T>>,
    r_parent: Option<&'a Tensor<'a, T>>
}

impl<'a, T: Num + Copy + Default> Tensor<'a, T> {
    fn new(data: Vec<T>, shape: &Vec<usize>) -> Tensor<'a, T> {
        let mut stride = vec![1];
        for i in 0..shape.len() - 1 {
            stride.push(stride[i] * shape[shape.len() - i - 1]);
        }
        stride.reverse();
        let len = data.len();
        Tensor {
            data: data,
            shape: shape.clone(),
            stride: stride,
            grad: vec![Default::default(); len],
            op: None,
            l_parent: None,
            r_parent: None
        }
    }
    fn new_parented(data: Vec<T>, shape: &Vec<usize>, op: TensOp, l_parent: Option<&'a Tensor<'a, T>>, r_parent: Option<&'a Tensor<'a, T>> ) -> Tensor<'a, T> {
        let mut stride = vec![1];
        for i in 0..shape.len() - 1 {
            stride.push(stride[i] * shape[shape.len() - i - 1]);
        }
        stride.reverse();
        let len = data.len();
        Tensor {
            data: data,
            shape: shape.clone(),
            stride: stride,
            grad: vec![Default::default(); len],
            op: Some(op),
            l_parent: l_parent,
            r_parent: r_parent
        }
    }
}
// rust macros are amazing
// now let's actually implement broadcasting properly
macro_rules! broadcast {
    ($b_trait:ident, $fn_name:ident) => {
        // I've been thinking about how to implement broadcasting in a nice abstract way, that generalizes
        // across binary operators
        // rust is :heart:
        impl<'a, T: Num + Copy + Default + AddAssign>$b_trait for Tensor<'a, T> {
            type Output = Tensor<'a, T>;
            fn $fn_name(self, rhs: Tensor<T>) -> Tensor<'a, T> {
                let mut res: Vec<T> = vec![];
                let mut grad: Vec<T> = vec![];
                let (smallest, largest) = if self.data.len() < rhs.data.len() {
                    (self, rhs)
                } else {
                    (rhs, self)
                };
                //let mut i = smallest.shape.len() - 1;
                //while smallest.shape[i] == largest.shape[i] && i >= 1 { i -= 1; }
                let stride = smallest.data.len();
                for i in 0..largest.data.len() {
                    res.push(largest.data[i].$fn_name(smallest.data[i % stride]));
                }
                return Tensor::new(res, &largest.shape);
            }
        }

        impl<'a, T: Num + Copy + Default + AddAssign> $b_trait<T> for Tensor<'a, T> {
            type Output = Tensor<'a, T>;
            fn $fn_name(self, rhs: T) -> Tensor<'a, T> {
                let mut res: Vec<T> = vec![];
                for i in 0..self.data.len() {
                    res.push(self.data[i].$fn_name(rhs));
                }
                return Tensor::new(res, &self.shape);
            }
        }
    }
}

// figure out how to specify base type constraints sheesh
// frustations with rust:
// - no type constraints on generics
// can't easily have a dynamic function name
// can't easily represent numerical types, should check out num
macro_rules! implement_reduce {
    ($fn_name:ident, $dim_fn:ident, $reduc_output:ident) => {
        impl<'a, T: Num + Copy + Default + AddAssign> Tensor<'a, T> where f64: From<T>, T: Div<Output = T> {
            fn $fn_name(self) -> $reduc_output {
                return $fn_name(&self.data);
            }
            fn $dim_fn(self, dim: usize) -> Tensor<'a, $reduc_output> {
                if dim >= self.shape.len() { panic!("Dimension too large. Tensor dims: {}, called dim: {}", self.shape.len(), dim); }
                // TODO: implement negative index later
                let mut res = vec![];
                // let's visualize this
                // dim = 0, easy
                // let's say we have a 3x2 tensor and we call dim = 1
                // we want it to become a (3,) tensor
                // data is (11, 12, 21, 22, 31, 32)
                // consider 2*2*2 tensor: (111, 112, 121, 122, 211, 212, 221, 222)
                // dim = 1 => stride = 4
                let op_step = if dim != 0 {
                    self.stride[dim-1]
                } else { self.data.len() };
                // rough draft of what could work, gotta think more
                for i in (0..self.data.len()).step_by(op_step) {
                    res.push($fn_name(&self.data[i..i+op_step]))
                }
                let shape = if dim == 0 {
                    Vec::from([1])
                } else { self.shape[0..dim].to_vec() };
                return Tensor::new(res, &shape);
            }
        }
    }
}

// this works, but need to figure out how to generalize this to all element wise functions
broadcast!(Add, add);
broadcast!(Mul, mul);
broadcast!(Div, div);
broadcast!(Sub, sub);


implement_reduce!{sum, sum_dim, T}
implement_reduce!{mean, mean_dim, f64}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, &shape);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.stride, vec![3, 1]);
    }

    #[test]
    fn test_tensor_creation_3d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![2, 2, 2];
        let tensor = Tensor::new(data, &shape);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(tensor.shape, vec![2, 2, 2]);
        assert_eq!(tensor.stride, vec![4, 2, 1]);
    }

    #[test]
    fn test_add() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = Tensor::new(data2, &shape);
        let tensor3 = tensor1 + tensor2;
        assert_eq!(tensor3.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_scalar() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = tensor1 + 1.0;
        assert_eq!(tensor2.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_mul_tensors() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = Tensor::new(data2, &shape);
        let tensor3 = tensor1 * tensor2;
        assert_eq!(tensor3.data, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    }

    #[test]
    fn test_div_tensors() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = Tensor::new(data2, &shape);
        let tensor3 = tensor1 / tensor2;
        assert_eq!(tensor3.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_broadcasting() {
        // [[1, 2, 3], [4, 5, 6]] * [1, 2, 3]
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let data2 = vec![1.0, 2.0];
        let shape1 = vec![2, 3, 2];
        let shape2 = vec![3];
        let tensor1 = Tensor::new(data1, &shape1);
        let tensor2 = Tensor::new(data2, &shape2);
        let tensor3 = tensor1 * tensor2;
        println!("{:?}", tensor3.data);
        assert_eq!(tensor3.data, vec![1.0, 4.0, 3.0, 8.0, 5.0, 12.0, 7.0, 16.0, 9.0, 20.0, 11.0, 24.0]);
    }

    #[test]
    fn test_mul_broadcast() {
        // [[1, 2, 3], [4, 5, 6]] * [1, 2, 3]
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0];
        let shape1 = vec![2, 3];
        let shape2 = vec![3];
        let tensor1 = Tensor::new(data1, &shape1);
        let tensor2 = Tensor::new(data2, &shape2);
        let tensor3 = tensor1 * tensor2;
        assert_eq!(tensor3.data, vec![1.0, 4.0, 9.0, 4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_tensor_sum() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let sum = tensor1.sum();
        assert_eq!(sum, 21.0); // ugh

    }

    #[test]
    fn test_tensor_mean() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let mean = tensor1.mean();
        assert_eq!(mean, 3.5); // ugh
    }

    #[test]
    fn test_tensor_sum_dim() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let sum = tensor1.sum_dim(1);
        assert_eq!(sum.data, vec![6.0, 15.0]);
    }

    #[test]
    fn test_tensor_mean_dim() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let sum = tensor1.mean_dim(1);
        assert_eq!(sum.data, vec![2.0, 5.0]);
    }

}


