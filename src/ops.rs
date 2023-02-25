use std::ops::{AddAssign};
use num_traits::Num;

#[derive(Clone, Debug)]
pub enum TensOp {
    Add,
    Mul,
    Div,
    Sub,
    MatMul
}

pub fn sum<T: Copy + Default + AddAssign>(slice: &[T]) -> T {
    let mut sum: T = Default::default();
    for i in slice.iter() {
        sum += i.clone() // is this bad practice?
    }
    return sum;
}

// think about efficiency
pub fn mean<T: Copy + Default + Num + AddAssign + Into<f64>>(slice: &[T]) -> f64 {

    let mut sum: T = Default::default();
    for i in slice.iter() {
        sum += i.clone() // is this bad practice?
    }
    let sum2: f64 = sum.into();
    return sum2 / slice.len() as f64;
}

pub fn add<T: Copy + Num>(x: T, y: T) -> T {
    return x + y;
}
pub fn mul<T: Copy + Num>(x: T, y: T) -> T {
    return x * y;
}
