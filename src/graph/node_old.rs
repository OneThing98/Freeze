use crate::node::{Ones, Zeros};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::RwLock;

#[derive(new, Clone, Copy)]
//old tensor system