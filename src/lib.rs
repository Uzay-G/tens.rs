use std::ops::{Div, Mul, Add, Sub, AddAssign, Neg, Index};
use crate::ops::*; 
use num_traits::Num;
use std::cell::RefCell;
use std::rc::Rc;

mod ops;

#[derive(Debug)]
enum TensErrors {
    NoGrad
}

// we have our tensor type
#[derive(Clone, Debug)]
struct Tensor<'a, T> {
    data: Vec<T>,
    grad: RefCell<Option<Rc<Tensor<'a, T>>>>,
    shape: Vec<usize>,
    stride: Vec<usize>,
    op: Option<TensOp>,
    l_parent: Option<&'a Tensor<'a, T>>,
    r_parent: Option<&'a Tensor<'a, T>>
}

impl<'a, T: Num + Copy + Default + Neg + Neg<Output = T> + std::fmt::Display> Tensor<'a, T> {
    fn new(data: Vec<T>, shape: &Vec<usize>) -> Tensor<'a, T> {
        let mut stride = vec![1];
        for i in 0..shape.len() - 1 {
            stride.push(stride[i] * shape[shape.len() - i - 1]);
        }
        stride.reverse();
        Tensor {
            data: data,
            shape: shape.clone(),
            stride: stride,
            grad: RefCell::new(None),
            op: None,
            l_parent: None,
            r_parent: None
        }
    }
    fn new_parented<'b>(data: Vec<T>, shape: &Vec<usize>, op: TensOp, l_parent: Option<&'b Tensor<'b, T>>, r_parent: Option<&'b Tensor<'b, T>> ) -> Tensor<'b, T> {
        let mut stride = vec![1];
        for i in 0..shape.len() - 1 {
            stride.push(stride[i] * shape[shape.len() - i - 1]);
        }
        stride.reverse();
        Tensor {
            data: data,
            shape: shape.clone(),
            stride: stride,
            grad: RefCell::new(None),
            op: Some(op),
            l_parent: l_parent,
            r_parent: r_parent
        }
    }

    fn transpose(&self) -> Tensor<'a, T> {
        let mut new_shape = self.shape.clone();
        let mut new_stride = self.stride.clone();
        new_shape.reverse();
        new_stride.reverse();
        let mut new_data = vec![Default::default(); self.data.len()];
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                println!("{} {}, {}", i, j, self.stride[0]);
                new_data[j * self.shape[0] + i] = self.data[i * self.shape[1] + j];
            }
        }
        Tensor::new(new_data, &new_shape)
    }
    fn backward(&self, deriv: Option<Tensor<'a, T>>) where <T as Neg>::Output: Mul<T> {
        let grad = deriv.unwrap_or(Tensor::ones(&self.shape));
        *self.grad.borrow_mut() = Some(Rc::new(grad.clone()));
        match (self.l_parent, self.r_parent) {
            (Some(lhs), Some(rhs)) => {
                match self.op.as_ref().unwrap() {
                    TensOp::Add => {
                        lhs.backward(Some(grad.clone()));
                        rhs.backward(Some(grad.clone()));
                    }
                    TensOp::Mul => {
                        let mut l_grad = vec![T::zero(); lhs.data.len()];
                        let mut r_grad = vec![T::zero(); rhs.data.len()];
                        for i in 0..self.data.len() {
                            l_grad[i] = grad.data[i] * rhs.data[i];
                            r_grad[i] = grad.data[i] * lhs.data[i];
                        }
                        // use tensor ops to make this cleaner
                        lhs.backward(Some(Tensor::new(l_grad, &lhs.shape)));
                        rhs.backward(Some(Tensor::new(r_grad, &rhs.shape)));
                    }
                    TensOp::Div => {
                        let mut l_grad = vec![T::zero(); lhs.data.len()];
                        let mut r_grad = vec![T::zero(); rhs.data.len()];
                        for i in 0..self.data.len() {
                            println!("rhs.data[i]: {}", rhs.data[i]);
                            l_grad[i] = grad.data[i] / rhs.data[i];
                            r_grad[i] = -grad.data[i] * lhs.data[i] / (rhs.data[i] * rhs.data[i]);
                        }
                        lhs.backward(Some(Tensor::new(l_grad, &lhs.shape)));
                        rhs.backward(Some(Tensor::new(r_grad, &rhs.shape)));
                    }
                    TensOp::Sub => {
                        lhs.backward(Some(Tensor::ones(&lhs.shape)));
                        rhs.backward(Some(Tensor::ones(&rhs.shape)));
                    }
                    TensOp::MatMul => {
                        //
                    }
                }
            }
            (None, _) => {}
            (_, None) => {}
        }
    }
    /*
    fn transpose(&self) -> Tensor<T> {
        let mut new_shape = self.shape.clone();
        new_shape.swap(0, 1);
        let mut new_data = vec![T::zero(); self.data.len()];
        let idx = 0;
        for i in 0..self.data.len() {
            for j in 0..self.shape.len() {
            }
            new_data[idx] = self.data[i];
        }
        Tensor::new(new_data, &new_shape)
    }
    */
    fn matmul(&'a self, rhs: &'a Tensor<'a, T>) -> Tensor<T> where T: Mul<T, Output = T> + Add<T, Output = T> + Copy {
        let mut data = vec![T::zero(); self.shape[0] * rhs.shape[1]];
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                for k in 0..self.shape[1] {
                    data[i * rhs.shape[1] + j] = data[i * rhs.shape[1] + j] + self.data[i * self.shape[1] + k] * rhs.data[k * rhs.shape[1] + j];
                }
            }
        }
        Tensor::new_parented(data, &vec![self.shape[0], rhs.shape[1]], TensOp::MatMul, Some(self), Some(rhs))
    }

    fn ones(shape: &Vec<usize>) -> Tensor<'a, T> {
        let mut data = vec![Default::default(); shape.iter().product()];
        for i in 0..data.len() {
            data[i] = T::one();
        }
        Tensor::new(data, shape)
    }
    fn zeros(shape: &Vec<usize>) -> Tensor<'a, T> {
        let mut data = vec![Default::default(); shape.iter().product()];
        for i in 0..data.len() {
            data[i] = T::zero();
        }
        Tensor::new(data, shape)
    }
    fn gradient(&self) -> Result<Rc<Tensor<'a, T>>, TensErrors> {
        // need help here, this probably isn't super clean
        return Ok(self.grad.borrow_mut().as_mut().ok_or(TensErrors::NoGrad)?.clone());
    }
}

impl<T> PartialEq for Tensor<'_, T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl<T> Index<Vec<usize>> for Tensor<'_, T> {
    type Output = T;
    fn index(&self, idx: Vec<usize>) -> &T {
        if idx.len() < self.stride.len() { panic!("Indexing error"); }
        let mut index = 0;
        for i in 0..self.shape.len() {
            index += self.stride[i] * idx[i];
        }
        return &self.data[index];
    }
}

// rust macros are amazing
// now let's actually implement broadcasting properly
macro_rules! broadcast {
    ($b_trait:ident, $fn_name:ident) => {
        // I've been thinking about how to implement broadcasting in a nice abstract way, that generalizes
        // across binary operators
        // rust is :heart:
        impl<'a, T: Neg<Output = T> + std::fmt::Display+ Num + Copy + Default + AddAssign + std::ops::Neg>$b_trait<&'a Tensor<'a, T>> for &'a Tensor<'a, T> {
            type Output = Tensor<'a, T>;
            fn $fn_name(self, rhs: &'a Tensor<'a, T>) -> Tensor<T> {
                let mut res: Vec<T> = vec![];
                let (smallest, largest, inverted) = if self.data.len() < rhs.data.len() {
                    (self, rhs, false)
                } else {
                    (rhs, self, true)
                };
                //let mut i = smallest.shape.len() - 1;
                //while smallest.shape[i] == largest.shape[i] && i >= 1 { i -= 1; }
                let stride = smallest.data.len();
                for i in 0..largest.data.len() {
                    if (inverted)
                    {
                        res.push(largest.data[i].$fn_name(smallest.data[i % stride]));
                    } else {
                        res.push(smallest.data[i % stride].$fn_name(largest.data[i]));
                    }
                }
                //return Tensor::new(res, &largest.shape);
                // TODO: make inversion cleaner here
                if (inverted) {
                    return Tensor::new_parented(res, &(largest.shape.clone()), TensOp::$b_trait, Some(largest), Some(smallest));
                }
                else {
                    return Tensor::new_parented(res, &(largest.shape.clone()), TensOp::$b_trait, Some(smallest), Some(largest));
                }
            }
        }

        impl<'a, T: Neg<Output = T> + std::fmt::Display + Num + Copy + Default + AddAssign + std::ops::Neg> $b_trait<T> for &'a Tensor<'a, T> {
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
        impl<'a, T: Num + Copy + Default + AddAssign + std::ops::Neg> Tensor<'a, T> where f64: From<T>, T: Div<Output = T>, T: Neg<Output = T>, T: std::fmt::Display {
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
        let tensor3 = &tensor1 + &tensor2;
        assert_eq!(tensor3.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_scalar() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = &tensor1 + 1.0;
        assert_eq!(tensor2.data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_mul_tensors() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = Tensor::new(data2, &shape);
        let tensor3 = &tensor1 * &tensor2;
        assert_eq!(tensor3.data, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    }

    #[test]
    fn test_div_tensors() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape);
        let tensor2 = Tensor::new(data2, &shape);
        let tensor3 = &tensor1 / &tensor2;
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
        let tensor3 = &tensor1 * &tensor2;
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
        let tensor3 = &tensor1 * &tensor2;
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

    #[test]
    fn test_backprop() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let tensor2 = &tensor1 + 1.0;
        println!("{:?}", tensor2.data);
        let tensor3 = &tensor1 * &tensor2;
        let tensor4 = &tensor3 / &tensor2;
        assert_eq!(tensor3.data, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0]);
        println!("{:?}", tensor4.r_parent.unwrap().data);
        tensor3.backward(None);
        // is there any better way to do this?
        assert_eq!(tensor1.gradient().unwrap().data, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(tensor2.gradient().unwrap().data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        tensor4.backward(None);
        assert_eq!(tensor3.gradient().unwrap().data, (&Tensor::new(vec![1.0], &vec![1]) / &tensor2).data);
    }


    #[test]
    fn test_basic_index() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        assert_eq!(tensor1[vec![1, 2]], 6.0);
    }

    #[test]
    fn test_transpose() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, &shape1);
        let tensor2 = tensor1.transpose();
        assert_eq!(tensor2.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matrix_mul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = vec![2, 3];
        let shape2 = vec![3, 2];
        let tensor1 = Tensor::new(data1, &shape1);
        let tensor2 = Tensor::new(data2, &shape2);
        let tensor3 = tensor1.matmul(&tensor2);
        assert_eq!(tensor3.data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
