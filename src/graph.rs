use smallvec::SmallVec;
use std::collections::HashMap;

pub trait ControlFlowGraph {
    type BlockId;

    fn succs(&self) -> SmallVec<BlockId>;
}

pub struct Dominators<T: PartialEq + PartialOrd> {
    idom: HashMap<T, Option<T>>
}
