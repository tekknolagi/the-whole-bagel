#![allow(non_upper_case_globals)]
#![allow(dead_code)]
use std::collections::HashMap;
use bit_set::BitSet;
mod hir_type;

macro_rules! define_id_type {
    ($prefix:expr, $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(usize);

        impl From<usize> for $name { fn from(id: usize) -> Self { $name(id) } }
        impl From<$name> for usize { fn from(id: $name) -> Self { id.0 } }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                write!(f, "{self}")
            }
        }
    }
}

define_id_type!(":", NameId);

// ===== Begin matklad string interning =====

#[derive(Default)]
pub struct Interner {
    map: HashMap<&'static str, NameId>,
    vec: Vec<&'static str>,
    buf: String,
    full: Vec<String>,
}

impl Interner {
    pub fn with_capacity(cap: usize) -> Interner {
        let cap = cap.next_power_of_two();
        Interner {
            map: HashMap::default(),
            vec: Vec::new(),
            buf: String::with_capacity(cap),
            full: Vec::new(),
        }
    }

    pub fn intern(&mut self, name: &str) -> NameId {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let name = unsafe { self.alloc(name) };
        let id = NameId(self.map.len());
        self.map.insert(name, id);
        self.vec.push(name);

        debug_assert!(self.lookup(id) == name);
        debug_assert!(self.intern(name) == id);

        id
    }

    pub fn lookup(&self, id: NameId) -> &'static str {
        self.vec[id.0]
    }

    unsafe fn alloc(&mut self, name: &str) -> &'static str {
        let cap = self.buf.capacity();
        if cap < self.buf.len() + name.len() {
            let new_cap = (cap.max(name.len()) + 1)
                .next_power_of_two();
            let new_buf = String::with_capacity(new_cap);
            let old_buf = std::mem::replace(&mut self.buf, new_buf);
            self.full.push(old_buf);
        }

        let interned = {
            let start = self.buf.len();
            self.buf.push_str(name);
            &self.buf[start..]
        };

        &*(interned as *const str)
    }
}

// ===== End matklad string interning =====

#[derive(Clone, PartialEq)]
struct TypedBitSet<T> {
    set: BitSet,
    phantom: std::marker::PhantomData<T>,
}

impl<T> TypedBitSet<T> where T: From<usize>, usize: From<T> {
    pub fn new() -> Self {
        Self { set: BitSet::new(), phantom: Default::default() }
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn one(item: T) -> Self {
        let mut set = BitSet::new();
        set.insert(item.into());
        Self { set, phantom: Default::default() }
    }

    pub fn union(&self, other: &Self) -> Self {
        Self { set: self.set.union(&other.set).collect(), phantom: Default::default() }
    }

    pub fn contains(&self, item: T) -> bool {
        self.set.contains(item.into())
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.set.insert(item.into())
    }

    pub fn as_vec(&self) -> Vec<T> {
        self.set.iter().map(|idx| idx.into()).collect()
    }

    pub fn get_single(&self) -> T {
        assert_eq!(self.len(), 1);
        self.set.iter().next().unwrap().into()
    }

    pub fn iter<'a>(&self) -> impl Iterator<Item = T> + '_ {
        self.set.iter().map(|idx| idx.into())
    }
}

mod hir {
    use crate::{TypedBitSet, NameId, Interner};
    use smallstr::SmallString;
    use smallvec::{smallvec, SmallVec};
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use crate::hir_type::*;

    pub struct Lexer<'a> {
        chars: std::iter::Peekable<std::str::Chars<'a>>,
    }

    type IdentString = SmallString::<[u8; 7]>;

    #[derive(Debug, PartialEq, Clone)]
    pub enum Token {
        Ident(IdentString),
        Print,
        Str(String),
        Int(isize),
        Bool(bool),
        Float(f64),
        Semicolon,
        Less,
        LessEqual,
        Greater,
        GreaterEqual,
        Equal,
        EqualEqual,
        Plus,
        Minus,
        Star,
        ForwardSlash,
        BackSlash,
        Bang,
        BangEqual,
        Comma,
        Dot,
        Or,
        And,
        Var,
        If,
        Else,
        While,
        For,
        Fun,
        Return,
        Class,
        Nil,
        LParen,
        RParen,
        LCurly,
        RCurly,
    }

    #[derive(Debug, PartialEq)]
    pub enum LexError {
        Eof,
        UnexpectedChar(char),
        UnterminatedStringLiteral,
    }

    const NUMBER_BASE: u32 = 10;
    impl<'a> Lexer<'a> {
        pub fn from_str(source: &'a str) -> Lexer<'a> {
            Lexer { chars: source.chars().peekable() }
        }

        pub fn next_token(&mut self) -> Result<Token, LexError> {
            loop {
                match self.chars.next() {
                    None => { return Err(LexError::Eof); }
                    Some(c) if c.is_whitespace() => { continue; }
                    Some(c) if c.is_alphabetic() || c == '_' => { return self.read_ident(c); }
                    Some(c) if c.is_digit(NUMBER_BASE) => { return self.read_int(c.to_digit(NUMBER_BASE).unwrap() as isize); }
                    Some('"') => { return self.read_string(); }
                    Some(';') => { return Ok(Token::Semicolon); }
                    Some('+') => { return Ok(Token::Plus); }
                    Some('-') => { return Ok(Token::Minus); }
                    Some('*') => { return Ok(Token::Star); }
                    Some('/') => {
                        if self.chars.peek() == Some(&'/') {
                            self.chars.next();
                            self.read_comment();
                            continue;
                        }
                        return Ok(Token::ForwardSlash);
                    }
                    Some('\\') => { return Ok(Token::BackSlash); }
                    Some(',') => { return Ok(Token::Comma); }
                    Some('.') => { return Ok(Token::Dot); }
                    Some('!') => {
                        return Ok(match self.chars.peek() {
                            Some('=') => { self.chars.next(); Token::BangEqual }
                            _ => Token::Bang,
                        });
                    }
                    Some('(') => { return Ok(Token::LParen); }
                    Some(')') => { return Ok(Token::RParen); }
                    Some('{') => { return Ok(Token::LCurly); }
                    Some('}') => { return Ok(Token::RCurly); }
                    Some('<') => {
                        return Ok(match self.chars.peek() {
                            Some('=') => { self.chars.next(); Token::LessEqual }
                            _ => Token::Less,
                        });
                    }
                    Some('>') => {
                        return Ok(match self.chars.peek() {
                            Some('=') => { self.chars.next(); Token::GreaterEqual }
                            _ => Token::Greater,
                        });
                    }
                    Some('=') => {
                        return Ok(match self.chars.peek() {
                            Some('=') => { self.chars.next(); Token::EqualEqual }
                            _ => Token::Equal,
                        });
                    }
                    Some(c) => { return Err(LexError::UnexpectedChar(c)); }
                }
            }
        }

        fn read_ident(&mut self, c: char) -> Result<Token, LexError> {
            let mut result: IdentString = "".into();
            result.push(c);
            loop {
                match self.chars.peek() {
                    Some(c) if c.is_alphabetic() || *c == '_' || c.is_digit(NUMBER_BASE) => result.push(*c),
                    _ => break,
                }
                self.chars.next();
            }
            Ok(
                if result == "print" { Token::Print }
                else if result == "nil" { Token::Nil }
                else if result == "true" { Token::Bool(true) }
                else if result == "false" { Token::Bool(false) }
                else if result == "or" { Token::Or }
                else if result == "and" { Token::And }
                else if result == "var" { Token::Var }
                else if result == "if" { Token::If }
                else if result == "else" { Token::Else }
                else if result == "while" { Token::While }
                else if result == "for" { Token::For }
                else if result == "fun" { Token::Fun }
                else if result == "return" { Token::Return }
                else if result == "class" { Token::Class }
                else { Token::Ident(result) }
            )
        }

        fn read_string(&mut self) -> Result<Token, LexError> {
            let mut result: String = "".into();
            loop {
                match self.chars.peek() {
                    None => { return Err(LexError::UnterminatedStringLiteral); }
                    Some('"') => { self.chars.next(); break; }
                    Some(c) => { result.push(*c); self.chars.next(); }
                }
            }
            Ok(Token::Str(result))
        }

        fn read_int(&mut self, mut result: isize) -> Result<Token, LexError> {
            loop {
                match self.chars.peek().and_then(|c| c.to_digit(NUMBER_BASE)).and_then(|d| Some(d as isize)) {
                    Some(d) => result = result * (NUMBER_BASE as isize) + d,
                    _ => break,
                }
                self.chars.next();
            }
            Ok(Token::Int(result))
        }

        fn read_comment(&mut self) {
            loop {
                match self.chars.next() {
                    None => break,
                    Some('\n') => break,
                    _ => continue,
                }
            }
        }
    }

    impl<'a> std::iter::Iterator for Lexer<'a> {
        type Item = Token;
        fn next(&mut self) -> Option<Token> {
            match self.next_token() {
                Err(LexError::Eof) => { None }
                Err(e) => { eprintln!("Lexer error: {e:?}"); None }
                Ok(token) => Some(token),
            }
        }
    }

    define_id_type!("v", InsnId);
    define_id_type!("bb", BlockId);
    define_id_type!("fn", FunId);
    define_id_type!("@", Slot);

    type InsnSet = TypedBitSet<InsnId>;
    type BlockSet = TypedBitSet<BlockId>;

    #[derive(Debug)]
    struct Block {
        phis: Vec<InsnId>,
        insns: Vec<InsnId>,
    }

    impl Block {
        fn new() -> Block {
            Block { phis: vec![], insns: vec![] }
        }
    }

    #[derive(Debug)]
    pub struct Function {
        name: NameId,
        entry: BlockId,
        params: Vec<InsnId>,
        insns: Vec<Insn>,
        insn_types: Vec<Type>,
        blocks: Vec<Block>,
        num_locals: usize,
    }

    impl Function {
        fn new(name: NameId) -> Function {
            Function { name, entry: BlockId(0), insns: vec![], insn_types: vec![], params: vec![], blocks: vec![Block::new()], num_locals: 0 }
        }

        fn find(&self, insn: InsnId) -> InsnId {
            let mut result = insn;
            loop {
                let insn = &self.insns[result.0];
                match (&insn.opcode, insn.operands.as_slice()) {
                    (Opcode::Identity, [it]) => result = *it,
                    _ => return result,
                }
            }
        }

        fn make_equal_to(&mut self, left: InsnId, right: InsnId) {
            let found = self.find(left);
            self.insns[found.0] = Insn { opcode: Opcode::Identity, operands: smallvec![right] };
        }

        fn new_block(&mut self) -> BlockId {
            let result = BlockId(self.blocks.len());
            self.blocks.push(Block::new());
            result
        }

        fn new_insn(&mut self, insn: Insn) -> InsnId {
            let result = InsnId(self.insns.len());
            self.insns.push(insn);
            self.insn_types.push(TEmpty);
            result
        }

        fn new_param(&mut self, idx: usize) -> InsnId {
            let insn = self.new_insn(Insn { opcode: Opcode::Param(idx), operands: smallvec![] });
            self.params.push(insn);
            insn
        }

        fn push_insn(&mut self, block: BlockId, opcode: Opcode, operands: Operands) -> InsnId {
            let is_phi = opcode == Opcode::Phi;
            let result = self.new_insn(Insn { opcode, operands });
            if is_phi {
                self.blocks[block.0].phis.push(result);
            } else {
                // TODO(max): Catch double terminators
                self.blocks[block.0].insns.push(result);
            }
            self.insn_types.push(TEmpty);
            result
        }

        fn is_terminated(&self, block: BlockId) -> bool {
            match self.blocks[block.0].insns.last().map(|insn| &self.insns[insn.0].opcode) {
                Some(op) => op.is_terminator(),
                _ => false,
            }
        }

        fn succs(&self, block: BlockId) -> Successors {
            match self.blocks[block.0].insns.last().map(|insn| &self.insns[insn.0].opcode) {
                Some(Opcode::Return) => smallvec![],
                Some(Opcode::CondBranch(iftrue, iffalse)) => smallvec![*iftrue, *iffalse],
                Some(Opcode::Branch(target)) => smallvec![*target],
                _ => smallvec![],
            }
        }

        fn rpo(&self) -> Vec<BlockId> {
            let mut visited = BlockSet::new();
            let mut result = vec![];
            self.po_from(self.entry, &mut result, &mut visited);
            result.reverse();
            result
        }

        fn po_from(&self, block: BlockId, result: &mut Vec<BlockId>, visited: &mut BlockSet) {
            if visited.contains(block) { return; }
            visited.insert(block);
            for succ in self.succs(block) {
                self.po_from(succ, result, visited);
            }
            result.push(block);
        }

        pub fn unbox_locals(&mut self) {
            // TODO(max): Whole-heap modeling with escapes
            // TODO(max): Take into account aliasing
            // TODO(max): Allocation sinking and removal
            fn join(left: &Vec<InsnSet>, right: &Vec<InsnSet>) -> Vec<InsnSet> {
                assert_eq!(left.len(), right.len());
                left.into_iter().zip(right).map(|(l, r)| l.union(r)).collect()
            }
            let empty_state = vec![InsnSet::new(); self.num_locals];
            let mut block_entry = vec![empty_state.clone(); self.blocks.len()];
            let rpo = self.rpo();
            loop {
                let mut changed = false;
                for block_id in &rpo {
                    let mut env: Vec<_> = block_entry[block_id.0].clone();
                    for insn_id in &self.blocks[block_id.0].insns {
                        let Insn { opcode, operands } = &self.insns[insn_id.0];
                        match opcode {
                            Opcode::Store(slot) => {
                                env[slot.0] = InsnSet::one(operands[1]);
                            }
                            Opcode::Load(slot) => {
                                env[slot.0] = InsnSet::one(*insn_id);
                            }
                            _ => {}
                        }
                    }
                    for succ in self.succs(*block_id) {
                        let new = join(&block_entry[succ.0], &env);
                        if block_entry[succ.0] != new {
                            block_entry[succ.0] = new;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            for block_id in &rpo {
                let mut env: Vec<_> = std::mem::take(&mut block_entry[block_id.0]);
                let insns = std::mem::take(&mut self.blocks[block_id.0].insns);
                let mut new_insns = vec![];
                for insn_id in insns {
                    let Insn { opcode, operands } = &self.insns[insn_id.0];
                    match opcode {
                        Opcode::Store(slot) => {
                            env[slot.0] = InsnSet::one(operands[1]);
                            new_insns.push(insn_id);
                        }
                        Opcode::Load(slot) => {
                            let operands = &env[slot.0];
                            match operands.len() {
                                0 => panic!("Should have at least one value"),
                                1 => {
                                    let replacement = operands.get_single();
                                    self.make_equal_to(insn_id, replacement);
                                    new_insns.push(replacement);
                                }
                                _ => {
                                    let phi = self.push_insn(*block_id, Opcode::Phi, operands.as_vec().into());
                                    self.make_equal_to(insn_id, phi);
                                    new_insns.push(phi);
                                }
                            }
                        }
                        _ => new_insns.push(insn_id),
                    }
                }
                self.blocks[block_id.0].insns = new_insns;
            }
        }

        fn type_of(&self, insn_id: InsnId) -> Type {
            let insn_id = self.find(insn_id);
            self.insn_types[insn_id.0]
        }

        fn is_a(&self, insn_id: InsnId, ty: Type) -> bool {
            self.type_of(insn_id).is_subtype(ty)
        }

        fn infer_type(&self, insn_id: InsnId) -> Type {
            let insn_id = self.find(insn_id);
            let insn = &self.insns[insn_id.0];
            match &insn.opcode {
                Opcode::Identity => panic!("should not see Identity after calling find()"),
                Opcode::Const(Value::Int(v)) if crate::runtime::twb_smallint_is_valid(*v) => TSmallInt,
                Opcode::Const(Value::Int(_)) => TLargeInt,
                Opcode::Const(Value::Float(_)) => TFloat,
                Opcode::Const(Value::Bool(_)) => TBool,
                Opcode::Const(Value::Str(_)) => TStr,
                Opcode::Const(Value::ObjectClass) => TClass,
                Opcode::Const(Value::Nil) => TNil,
                Opcode::Abort => TVoid,
                Opcode::Print => TVoid,
                Opcode::Return | Opcode::Branch(_) | Opcode::CondBranch(_, _) => TVoid,
                Opcode::Store(_) => TVoid,
                Opcode::Add if self.is_a(insn.operands[0], TInt) && self.is_a(insn.operands[1], TInt) => TInt,
                Opcode::Add if self.is_a(insn.operands[0], TFloat) && self.is_a(insn.operands[1], TFloat) => TFloat,
                Opcode::Add if self.is_a(insn.operands[0], TStr) && self.is_a(insn.operands[1], TStr) => TStr,
                Opcode::IsTruthy => TCBool,
                Opcode::NewFrame => TFrame,
                Opcode::NewClass(_) => TClass,
                Opcode::GuardInt => TInt,
                Opcode::Less | Opcode::LessEqual | Opcode::Greater | Opcode::GreaterEqual => TBool,
                _ => TAny,
            }
        }

        pub fn infer_types(&mut self) {
            let mut changed = true;
            let rpo = self.rpo();
            while changed {
                changed = false;
                for block_id in &rpo {
                    for insn_id in &self.blocks[block_id.0].insns {
                        let insn_id = self.find(*insn_id);
                        let old_type = self.insn_types[insn_id.0];
                        let new_type = self.infer_type(insn_id);
                        if !old_type.bit_equal(new_type) {
                            self.insn_types[insn_id.0] = new_type;
                            changed = true;
                        }
                    }
                }
            }
        }

        fn is_critical(&self, insn_id: InsnId) -> bool {
            let insn_id = self.find(insn_id);
            match &self.insns[insn_id.0].opcode {
                Opcode::Const(_) => false,
                Opcode::Abort => true,
                Opcode::Print => true,
                Opcode::Return => true,
                Opcode::Add => true,
                Opcode::Sub => true,
                Opcode::Mul => true,
                Opcode::Div => true,
                Opcode::Equal => false,
                Opcode::NotEqual => false,
                Opcode::Greater => true,
                Opcode::GreaterEqual => true,
                Opcode::Less => true,
                Opcode::LessEqual => true,
                Opcode::Param(_) => false,
                Opcode::Branch(_) => true,
                Opcode::CondBranch(..) => true,
                Opcode::Phi => false,
                Opcode::NewFrame => false,
                Opcode::Load(_) => false,
                Opcode::Store(_) => true,
                Opcode::LoadAttr(_) => true,
                Opcode::StoreAttr(_) => true,
                Opcode::GuardInt => true,
                Opcode::GuardClass => true,
                Opcode::GuardInstance => true,
                Opcode::IsTruthy => false,
                Opcode::NewClass(_) => false,
                Opcode::This => false,
                Opcode::NewClosure(_) => false,
                Opcode::Call => true,
                Opcode::Identity => panic!("Should not see Identity after calling find()"),
            }
        }

        pub fn eliminate_dead_code(&mut self) {
            let mut worklist = VecDeque::new();
            let rpo = self.rpo();
            for block_id in &rpo {
                for insn_id in &self.blocks[block_id.0].insns {
                    let insn_id = self.find(*insn_id);
                    if self.is_critical(insn_id) {
                        worklist.push_back(insn_id);
                    }
                }
            }
            let mut mark = vec![false; self.insns.len()];
            while let Some(insn) = worklist.pop_front() {
                if mark[insn.0] { continue; }
                mark[insn.0] = true;
                let insn_id = self.find(insn);
                for operand in &self.insns[insn_id.0].operands {
                    let operand = self.find(*operand);
                    worklist.push_back(operand);
                }
            }
            for block_id in &rpo {
                let old_block = &self.blocks[block_id.0].insns;
                let mut new_block = vec![];
                for insn_id in old_block {
                    let insn_id = self.find(*insn_id);
                    if mark[insn_id.0] {
                        new_block.push(insn_id);
                    }
                }
                self.blocks[block_id.0].insns = new_block;
            }
        }
    }

    struct Dominators {
        idom: HashMap<BlockId, BlockId>,
        df: HashMap<BlockId, BlockSet>,
    }

    impl Dominators {
        // From "A Simple, Fast Dominance Algorithm" by Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
        fn new(fun: &Function) -> Self {
            // All nodes start as Undefined because the HashMap is empty
            let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
            idom.insert(fun.entry, fun.entry);
            let rpo = fun.rpo();
            // Compute preds
            let mut preds: HashMap<BlockId, BlockSet> = HashMap::new();
            for block_id in &rpo {
                for succ in fun.succs(*block_id) {
                    preds.entry(succ)
                        .and_modify(|succ_preds| { succ_preds.insert(*block_id); })
                        .or_insert_with(|| BlockSet::one(*block_id));
                }
            }
            // Compute immediate dominators
            assert_eq!(rpo[0], fun.entry);
            let mut changed = true;
            while changed {
                changed = false;
                // Skip the entry (it's in rpo[0]; see above)
                for block_id in &rpo[1..] {
                    // Find the first processed predecessor
                    let first_processed = preds[block_id].iter().take_while(|p| !idom.contains_key(p)).next().unwrap();
                    let mut new_idom = first_processed;
                    // For all other predecessors with an entry in idom
                    for pred in preds[block_id].iter().filter(|p| *p != first_processed && idom.contains_key(p)) {
                        // Intersect pred, new_idom
                        let mut finger1 = pred;
                        let mut finger2 = new_idom;
                        while finger1 != finger2 {
                            while finger1 < finger2 {
                                finger1 = idom[&finger1];
                            }
                            while finger2 < finger1 {
                                finger2 = idom[&finger2];
                            }
                        }
                        new_idom = finger1;
                    }
                    if idom[block_id] != new_idom {
                        idom.insert(*block_id, new_idom);
                        changed = true;
                    }
                }
            }
            // Compute dominance frontiers
            let mut df: HashMap<BlockId, BlockSet> = HashMap::new();
            for block_id in &rpo {
                if preds[block_id].len() >= 2 {
                    for pred in preds[block_id].iter() {
                        let mut runner = pred;
                        while runner != idom[block_id] {
                            df.entry(runner)
                                .and_modify(|df| { df.insert(*block_id); })
                                .or_insert_with(|| BlockSet::one(*block_id));
                            runner = idom[&runner];
                        }
                    }
                }
            }
            Self { idom, df }
        }
    }

    struct FunctionPrinter<'a> {
        program: &'a Program,
        function: &'a Function,
    }

    impl<'a> FunctionPrinter<'a> {
        fn new(program: &'a Program, function: &'a Function) -> Self {
            Self { program, function }
        }
    }

    impl<'a> std::fmt::Display for FunctionPrinter<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            let fun = self.function;
            let fun_name = self.program.interner.lookup(fun.name);
            let fun_entry = fun.entry;
            write!(f, "fun {fun_name}(")?;
            let mut sep = "";
            for param in &fun.params {
                write!(f, "{sep}{:?}", fun.find(*param))?;
                sep = ", ";
            }
            writeln!(f, ") (entry {fun_entry}) {{")?;
            let mut seen = InsnSet::new();
            for block_id in fun.rpo() {
                writeln!(f, "  {block_id} {{")?;
                let block = &fun.blocks[block_id.0];
                for insn_id in block.phis.iter().chain(&block.insns) {
                    let insn_id = fun.find(*insn_id);
                    if seen.contains(insn_id) { continue; }
                    seen.insert(insn_id);
                    let Insn { opcode, operands } = &fun.insns[insn_id.0];
                    let ty = fun.type_of(insn_id);
                    if ty.bit_equal(TEmpty) {
                        write!(f, "    {insn_id} = ")?;
                    } else if ty.bit_equal(TVoid) {
                        write!(f, "    ")?;
                    } else {
                        write!(f, "    {insn_id}:{ty} = ")?;
                    }
                    match opcode {
                        Opcode::NewClass(ClassDef { name, superclass, .. }) => {
                            let class_name = self.program.interner.lookup(*name);
                            write!(f, "NewClass({class_name}, {superclass})")
                        }
                        Opcode::LoadAttr(name) => {
                            let name = self.program.interner.lookup(*name);
                            write!(f, "LoadAttr({name})")
                        }
                        Opcode::StoreAttr(name) => {
                            let name = self.program.interner.lookup(*name);
                            write!(f, "StoreAttr({name})")
                        }
                        _ => write!(f, "{:?}", opcode),
                    }?;
                    let mut sep = "";
                    for operand in operands {
                        write!(f, "{sep} {:?}", fun.find(*operand))?;
                        sep = ",";
                    }
                    write!(f, "\n")?;
                }
                writeln!(f, "  }}")?;
            }
            writeln!(f, "}}")
        }
    }

    #[derive(Debug, PartialEq, Clone)]
    enum Value {
        Nil,
        Int(isize),
        Float(f64),
        Bool(bool),
        Str(String),
        ObjectClass,
    }

    #[derive(Debug, PartialEq)]
    struct ClassDef {
        name: NameId,
        superclass: InsnId,
        methods: Vec<FunId>,
    }

    #[derive(Debug, PartialEq)]
    enum Opcode {
        Identity,
        Const(Value),
        Abort,
        Print,
        Return,
        Add,
        Sub,
        Mul,
        Div,
        Equal,
        NotEqual,
        Greater,
        GreaterEqual,
        Less,
        LessEqual,
        Param(usize),
        Branch(BlockId),
        CondBranch(BlockId, BlockId),
        Phi,
        NewFrame,
        NewClass(ClassDef),
        NewClosure(FunId),
        Load(Slot),
        Store(Slot),
        LoadAttr(NameId),
        StoreAttr(NameId),
        GuardInt,
        GuardClass,
        GuardInstance,
        IsTruthy,
        This,
        Call,
    }

    impl Opcode {
        fn is_terminator(&self) -> bool {
            match self {
                Opcode::Return | Opcode::Branch(..) | Opcode::CondBranch(..) => true,
                _ => false,
            }
        }
    }

    type Operands = SmallVec::<[InsnId; 2]>;
    type Successors = SmallVec::<[BlockId; 2]>;

    #[derive(Debug)]
    struct Insn {
        opcode: Opcode,
        operands: Operands,
    }

    pub struct Program {
        entry: FunId,
        // TODO(max): Make an API for iterating over Function instead of making the field public
        pub funs: Vec<Function>,
        interner: Interner,
    }

    #[derive(PartialEq)]
    enum Assoc {
        Any,
        Left,
        Right
    }

    impl Program {
        fn new() -> Program {
            let mut result = Program { entry: FunId(0), funs: vec![Function::new(NameId(0))], interner: Default::default() };
            let name = result.intern("<toplevel>");
            result.funs[0].name = name;
            result
        }

        fn intern(&mut self, name: &str) -> NameId {
            self.interner.intern(name)
        }

        fn push_fun(&mut self, name: NameId) -> FunId {
            let fun = FunId(self.funs.len());
            self.funs.push(Function::new(name));
            fun
        }

        fn push_insn(&mut self, fun: FunId, block: BlockId, opcode: Opcode, operands: Operands) -> InsnId {
            self.funs[fun.0].push_insn(block, opcode, operands)
        }
    }

    impl std::fmt::Display for Program {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            writeln!(f, "Entry: {}", self.entry)?;
            for (idx, fun) in self.funs.iter().enumerate() {
                let fun_id = FunId(idx);
                let printer = FunctionPrinter::new(self, fun);
                write!(f, "{fun_id}: {printer}")?;
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Env<'a> {
        fun: FunId,
        bindings: HashMap<NameId, Slot>,
        parent: Option<&'a Env<'a>>,
    }

    impl<'a> Env<'a> {
        fn new(fun: FunId) -> Env<'a> {
            Env { fun, bindings: HashMap::new(), parent: None }
        }

        fn from_parent(fun: FunId, parent: &'a Env<'a>) -> Env<'a> {
            Env { fun, bindings: HashMap::new(), parent: Some(parent) }
        }

        fn lookup(&self, name: NameId) -> Option<Slot> {
            self.bindings.get(&name).copied()
        }

        fn define(&mut self, name: NameId) -> Slot {
            let slot = Slot(self.bindings.len());
            self.bindings.insert(name.clone(), slot);
            slot
        }
    }

    struct Context {
        fun: FunId,
        block: BlockId,
        frame: InsnId,
    }

    pub struct Parser<'a> {
        tokens: std::iter::Peekable<&'a mut Lexer<'a>>,
        pub prog: Program,
        context_stack: Vec<Context>,
    }

    #[derive(Debug, PartialEq)]
    pub enum ParseError {
        UnexpectedToken(Token),
        UnexpectedEof,
        UnboundName(&'static str),
        VariableShadows(&'static str),
        CannotAssignTo(LValue),
        CannotAssignToThis,
        CannotInheritSelf(&'static str),
    }

    #[derive(Debug, PartialEq)]
    pub enum LValue {
        Insn(InsnId),
        Name(NameId),
        Attr(InsnId, NameId),
    }

    impl Parser<'_> {
        pub fn from_lexer<'a>(lexer: &'a mut Lexer<'a>) -> Parser<'a> {
            Parser { tokens: lexer.peekable(), prog: Program::new(), context_stack: vec![] }
        }

        fn enter_fun(&mut self, fun: FunId) {
            let block = self.prog.funs[fun.0].entry;
            let frame = self.prog.push_insn(fun, block, Opcode::NewFrame, smallvec![]);
            self.context_stack.push(Context { fun, block, frame });
            self.enter_block(block);
        }

        fn leave_fun(&mut self) {
            let fun = self.fun();
            if !self.prog.funs[fun.0].is_terminated(self.block()) {
                let nil = self.push_op(Opcode::Const(Value::Nil));
                self.push_insn(Opcode::Return, smallvec![nil]);
            }
            self.prog.funs[fun.0].infer_types();
            if let Some(Context { .. }) = self.context_stack.pop() {
            } else {
                panic!("Function stack underflow");
            }
        }

        fn fun(&self) -> FunId {
            self.context_stack.last().expect("Function stack underflow").fun
        }

        fn block(&self) -> BlockId {
            self.context_stack.last().expect("Function stack underflow").block
        }

        fn frame(&self) -> InsnId {
            self.context_stack.last().expect("Function stack underflow").frame
        }

        fn new_block(&mut self) -> BlockId {
            let fun = self.fun();
            self.prog.funs[fun.0].new_block()
        }

        fn enter_block(&mut self, block_id: BlockId) {
            self.context_stack.last_mut().unwrap().block = block_id;
        }

        fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
            match self.tokens.next() {
                None => Err(ParseError::UnexpectedEof),
                Some(actual) if expected == actual => Ok(()),
                Some(actual) => Err(ParseError::UnexpectedToken(actual)),
            }
        }

        fn expect_ident(&mut self) -> Result<NameId, ParseError> {
            match self.tokens.next() {
                Some(Token::Ident(name)) => Ok(self.prog.intern(&name)),
                None => Err(ParseError::UnexpectedEof),
                Some(actual) => Err(ParseError::UnexpectedToken(actual)),
            }
        }

        fn push_insn(&mut self, opcode: Opcode, operands: Operands) -> InsnId {
            self.prog.push_insn(self.fun(), self.block(), opcode, operands)
        }

        fn push_op(&mut self, opcode: Opcode) -> InsnId {
            self.prog.push_insn(self.fun(), self.block(), opcode, smallvec![])
        }

        pub fn parse_program(&mut self) -> Result<(), ParseError> {
            self.enter_fun(self.prog.entry);
            let mut env = Env::new(self.prog.entry);
            while let Some(_) = self.tokens.peek() {
                self.parse_toplevel(&mut env)?;
            }
            self.leave_fun();
            Ok(())
        }

        fn parse_class(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
            self.expect(Token::Class)?;
            let name = self.expect_ident()?;
            let superclass = match self.tokens.peek() {
                Some(Token::Less) => {
                    self.tokens.next();
                    let superclass_name = self.expect_ident()?;
                    if name == superclass_name {
                        return Err(ParseError::CannotInheritSelf(self.prog.interner.lookup(name)));
                    }
                    let value = self.load_local(env, superclass_name)?;
                    self.push_insn(Opcode::GuardClass, smallvec![value])
                }
                _ => self.push_op(Opcode::Const(Value::ObjectClass)),
            };
            self.expect(Token::LCurly)?;
            let mut methods = vec![];
            loop {
                if let Some(Token::RCurly) = self.tokens.peek() {
                    break;
                }
                let method = self.parse_method(&mut env)?;
                methods.push(method);
            }
            self.expect(Token::RCurly)?;
            let class = self.push_op(Opcode::NewClass(ClassDef { name, superclass, methods }));
            self.define_local(&mut env, name, class);
            Ok(())
        }

        fn parse_method(&mut self, env: &Env) -> Result<FunId, ParseError> {
            let name = self.expect_ident()?;
            let fun = self.prog.push_fun(name);
            self.enter_fun(fun);
            self.expect(Token::LParen)?;
            let mut idx = 0;
            // TODO(max): We need a way to find out if a variable is from an outer context (global,
            // closure) when using it so we don't bake in the value at compile-time.
            let mut func_env = Env::from_parent(fun, &env);
            let this_name = self.prog.intern("this");
            // TODO(max): Decide if this should be its own opcode or just another param
            let this = self.prog.funs[fun.0].new_insn(Insn { opcode: Opcode::This, operands: smallvec![] });
            self.prog.funs[fun.0].params.push(this);
            self.define_local(&mut func_env, this_name, this);
            loop {
                match self.tokens.peek() {
                    Some(Token::Ident(name)) => {
                        let name = self.prog.intern(name);
                        let param = self.prog.funs[fun.0].new_param(idx);
                        self.define_local(&mut func_env, name, param);
                        self.tokens.next();
                        idx += 1;
                    }
                    _ => break,
                }
                match self.tokens.peek() {
                    Some(Token::Comma) => { self.tokens.next(); }
                    _ => break,
                }
            }
            self.expect(Token::RParen)?;
            self.expect(Token::LCurly)?;
            while let Some(token) = self.tokens.peek() {
                if *token == Token::RCurly { break; }
                self.parse_statement(&mut func_env)?;
            }
            self.expect(Token::RCurly)?;
            self.leave_fun();
            Ok(fun)
        }

        fn parse_toplevel(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
            match self.tokens.peek() {
                _ => self.parse_statement(&mut env),
            }
        }

        fn parse_statement(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
            match self.tokens.peek() {
                Some(Token::Print) => {
                    self.tokens.next();
                    let expr = self.parse_expression(&mut env)?;
                    self.push_insn(Opcode::Print, smallvec![expr]);
                    Ok(())
                }
                Some(Token::Return) => {
                    self.tokens.next();
                    let expr = self.parse_expression(&mut env)?;
                    self.push_insn(Opcode::Return, smallvec![expr]);
                    Ok(())
                }
                Some(Token::Fun) => {
                    self.tokens.next();
                    return self.parse_function(&mut env);  // no semicolon
                }
                Some(Token::Class) => return self.parse_class(&mut env),  // no semicolon
                Some(Token::Var) => {
                    self.tokens.next();
                    let name = self.expect_ident()?;
                    self.expect(Token::Equal)?;
                    let value = self.parse_expression(&mut env)?;
                    self.define_local(&mut env, name, value);
                    Ok(())
                }
                Some(Token::If) => {
                    self.tokens.next();
                    self.expect(Token::LParen)?;
                    let cond = self.parse_expression(&mut env)?;
                    let cond = self.push_insn(Opcode::IsTruthy, smallvec![cond]);
                    self.expect(Token::RParen)?;
                    let iftrue_block = self.new_block();
                    let iffalse_block = self.new_block();
                    self.push_insn(Opcode::CondBranch(iftrue_block, iffalse_block), smallvec![cond]);

                    self.enter_block(iftrue_block);
                    let mut iftrue_env = env.clone();
                    self.parse_statement(&mut iftrue_env)?;
                    let iftrue_end = self.block();

                    self.enter_block(iffalse_block);
                    if self.tokens.peek() == Some(&Token::Else) {
                        self.tokens.next();
                        let mut iffalse_env = env.clone();
                        self.parse_statement(&mut iffalse_env)?;
                        let join_block = self.new_block();
                        self.push_op(Opcode::Branch(join_block));

                        self.enter_block(join_block);
                        self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(join_block), smallvec![]);
                    } else {
                        self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(iffalse_block), smallvec![]);
                    }
                    return Ok(());  // no semicolon
                }
                Some(Token::While) => {
                    self.tokens.next();
                    self.expect(Token::LParen)?;
                    let header_block = self.new_block();
                    self.push_op(Opcode::Branch(header_block));
                    self.enter_block(header_block);
                    let cond = self.parse_expression(&mut env)?;
                    let cond = self.push_insn(Opcode::IsTruthy, smallvec![cond]);
                    self.expect(Token::RParen)?;
                    let body_block = self.new_block();
                    let after_block = self.new_block();
                    self.push_insn(Opcode::CondBranch(body_block, after_block), smallvec![cond]);
                    self.enter_block(body_block);
                    let mut body_env = env.clone();
                    self.parse_statement(&mut body_env)?;
                    self.push_op(Opcode::Branch(header_block));
                    self.enter_block(after_block);
                    return Ok(());  // no semicolon
                }
                Some(Token::LCurly) => {
                    // New scope
                    let mut block_env = env.clone();
                    self.tokens.next();
                    while let Some(token) = self.tokens.peek() {
                        if *token == Token::RCurly { break; }
                        self.parse_statement(&mut block_env)?;
                    }
                    return self.expect(Token::RCurly);  // no semicolon
                }
                Some(_) => { self.parse_expression(&mut env)?; Ok(()) },
                None => { Err(ParseError::UnexpectedEof) }
            }?;
            self.expect(Token::Semicolon)
        }

        fn write_local(&mut self, slot: Slot, value: InsnId) {
            let fun_id = self.fun();
            let num_locals = &mut self.prog.funs[fun_id.0].num_locals;
            *num_locals = std::cmp::max(*num_locals, slot.0) + 1;
            self.push_insn(Opcode::Store(slot), smallvec![self.frame(), value]);
        }

        fn read_local(&mut self, slot: Slot) -> InsnId {
            let fun_id = self.fun();
            let num_locals = &mut self.prog.funs[fun_id.0].num_locals;
            *num_locals = std::cmp::max(*num_locals, slot.0) + 1;
            self.push_insn(Opcode::Load(slot), smallvec![self.frame()])
        }

        fn parse_function(&mut self, env: &mut Env) -> Result<(), ParseError> {
            let name = self.expect_ident()?;
            let fun = self.prog.push_fun(name.clone());
            self.enter_fun(fun);
            self.expect(Token::LParen)?;
            let mut idx = 0;
            // TODO(max): We need a way to find out if a variable is from an outer context (global,
            // closure) when using it so we don't bake in the value at compile-time.
            let mut func_env = Env::from_parent(fun, &env);
            loop {
                match self.tokens.peek() {
                    Some(Token::Ident(name)) => {
                        let name = self.prog.intern(name);
                        let param = self.prog.funs[fun.0].new_param(idx);
                        self.define_local(&mut func_env, name, param);
                        self.tokens.next();
                        idx += 1;
                    }
                    _ => break,
                }
                match self.tokens.peek() {
                    Some(Token::Comma) => { self.tokens.next(); }
                    _ => break,
                }
            }
            self.expect(Token::RParen)?;
            self.expect(Token::LCurly)?;
            while let Some(token) = self.tokens.peek() {
                if *token == Token::RCurly { break; }
                self.parse_statement(&mut func_env)?;
            }
            self.expect(Token::RCurly)?;
            self.leave_fun();
            let closure = self.push_op(Opcode::NewClosure(fun));
            self.define_local(env, name, closure);
            Ok(())
        }

        fn parse_args(&mut self, mut env: &mut Env, args: &mut Operands) -> Result<(), ParseError> {
            // TODO(max): Come up with a better idiom to parse comma-separated lists. This is wack.
            // TODO(max): Disallow f(x,)
            loop {
                match self.tokens.peek() {
                    Some(Token::RParen) => break,
                    Some(Token::Comma) => return Err(ParseError::UnexpectedToken(Token::Comma)),
                    None => return Err(ParseError::UnexpectedEof),
                    _ => args.push(self.parse_expression(&mut env)?),
                }
                match self.tokens.peek() {
                    Some(Token::Comma) => { self.tokens.next(); }
                    _ => break,
                }
            }
            Ok(())
        }

        fn parse_expression(&mut self, env: &mut Env) -> Result<InsnId, ParseError> {
            self.parse_(env, 0)
        }

        fn load_local(&mut self, env: &Env, name: NameId) -> Result<InsnId, ParseError> {
            match env.lookup(name) {
                Some(slot) => Ok(self.read_local(slot)),
                None => return Err(ParseError::UnboundName(self.prog.interner.lookup(name))),
            }
        }

        fn define_local(&mut self, env: &mut Env, name: NameId, value: InsnId) {
            let slot = env.define(name);
            self.write_local(slot, value);
        }

        fn store_local(&mut self, env: &Env, name: NameId, value: InsnId) -> Result<InsnId, ParseError> {
            if name == self.prog.intern("this") {
                return Err(ParseError::CannotAssignToThis);
            }
            match env.lookup(name) {
                Some(slot) => { self.write_local(slot, value); Ok(value) }
                None => return Err(ParseError::UnboundName(self.prog.interner.lookup(name))),
            }
        }

        fn lvalue_as_rvalue(&mut self, env: &Env, lvalue: LValue) -> Result<InsnId, ParseError> {
            match lvalue {
                LValue::Insn(insn_id) => Ok(insn_id),
                LValue::Name(name) => self.load_local(env, name),
                LValue::Attr(obj, name) => {
                    let obj = self.push_insn(Opcode::GuardInstance, smallvec![obj]);
                    Ok(self.push_insn(Opcode::LoadAttr(name), smallvec![obj]))
                }
            }
        }

        fn parse_(&mut self, mut env: &mut Env, prec: u32) -> Result<InsnId, ParseError> {
            let mut lhs = match self.tokens.next() {
                None => return Err(ParseError::UnexpectedEof),
                Some(Token::Nil) => {
                    LValue::Insn(self.push_op(Opcode::Const(Value::Nil)))
                }
                Some(Token::Bool(value)) => {
                    LValue::Insn(self.push_op(Opcode::Const(Value::Bool(value))))
                }
                Some(Token::Int(value)) => {
                    LValue::Insn(self.push_op(Opcode::Const(Value::Int(value))))
                }
                Some(Token::Str(value)) => {
                    LValue::Insn(self.push_op(Opcode::Const(Value::Str(value))))
                }
                Some(Token::Ident(name)) => {
                    LValue::Name(self.prog.intern(&name))
                }
                Some(Token::LParen) => {
                    let result = self.parse_(&mut env, 0)?;
                    self.expect(Token::RParen)?;
                    LValue::Insn(result)
                }
                Some(token) => return Err(ParseError::UnexpectedToken(token)),
            };
            while let Some(token) = self.tokens.peek() {
                let (assoc, op_prec) = match token {
                    Token::Equal => (Assoc::Right, 0),
                    Token::And => (Assoc::Left, 1),
                    Token::Or => (Assoc::Left, 1),
                    Token::EqualEqual => (Assoc::Left, 2),
                    Token::BangEqual => (Assoc::Left, 2),
                    Token::Greater => (Assoc::Left, 3),
                    Token::GreaterEqual => (Assoc::Left, 3),
                    Token::Less => (Assoc::Left, 3),
                    Token::LessEqual => (Assoc::Left, 3),
                    Token::Plus => (Assoc::Any, 4),
                    Token::Minus => (Assoc::Left, 4),
                    Token::Star => (Assoc::Any, 5),
                    Token::ForwardSlash => (Assoc::Left, 5),
                    Token::LParen => (Assoc::Any, 6),
                    // TODO(max): Check associativity of .
                    Token::Dot => (Assoc::Left, 7),
                    _ => break,
                };
                let token = token.clone();
                if op_prec < prec { return self.lvalue_as_rvalue(env, lhs); }
                self.tokens.next();
                let next_prec = if assoc == Assoc::Left { op_prec + 1 } else { op_prec };
                // Special case: we need to look up the frame slot and write to it or assign to the
                // attribute
                if token == Token::Equal {
                    lhs = match lhs {
                        LValue::Insn(..) => return Err(ParseError::CannotAssignTo(lhs)),
                        LValue::Name(name) => {
                            let rhs = self.parse_(&mut env, next_prec)?;
                            self.store_local(&mut env, name, rhs)?;
                            lhs
                        }
                        LValue::Attr(obj, name) => {
                            let rhs = self.parse_(&mut env, next_prec)?;
                            let obj = self.push_insn(Opcode::GuardInstance, smallvec![obj]);
                            self.push_insn(Opcode::StoreAttr(name), smallvec![obj, rhs]);
                            LValue::Insn(rhs)
                        }
                    };
                    continue;
                }
                // Special case: we need to hold off evaluating the attribute until it is read
                // elsewhere or written
                if token == Token::Dot {
                    let name = self.expect_ident()?;
                    let obj = self.lvalue_as_rvalue(env, lhs)?;
                    lhs = LValue::Attr(obj, name);
                    continue;
                }
                // Evaluate the lvalue because we want to use it in a binary operation
                let mut lhs_value = self.lvalue_as_rvalue(env, lhs)?;
                if token == Token::And {
                    todo!("ssa from `and' keyword")
                }
                if token == Token::Or {
                    let iftrue_block = self.new_block();
                    let iffalse_block = self.new_block();
                    let lhs_truthy = self.push_insn(Opcode::IsTruthy, smallvec![lhs_value]);
                    self.push_insn(Opcode::CondBranch(iftrue_block, iffalse_block), smallvec![lhs_truthy.clone()]);
                    self.enter_block(iffalse_block);
                    let rhs = self.parse_(&mut env, next_prec)?;
                    self.push_op(Opcode::Branch(iftrue_block));
                    self.enter_block(iftrue_block);
                    lhs = LValue::Insn(self.push_insn(Opcode::Phi, smallvec![lhs_value, rhs]));
                    continue;
                }
                if token == Token::LParen {
                    // Function call
                    let mut operands = smallvec![lhs_value];
                    self.parse_args(&mut env, &mut operands)?;
                    self.expect(Token::RParen)?;
                    lhs = LValue::Insn(self.push_insn(Opcode::Call, operands));
                    continue;
                }
                let opcode = match token {
                    Token::EqualEqual => Opcode::Equal,
                    Token::BangEqual => Opcode::NotEqual,
                    Token::Greater => Opcode::Greater,
                    Token::GreaterEqual => Opcode::GreaterEqual,
                    Token::Less => Opcode::Less,
                    Token::LessEqual => Opcode::LessEqual,
                    Token::Plus => Opcode::Add,
                    Token::Minus => Opcode::Sub,
                    Token::Star => Opcode::Mul,
                    Token::ForwardSlash => Opcode::Div,
                    _ => panic!("Unexpected token {token:?}"),
                };
                let mut rhs = self.parse_(&mut env, next_prec)?;
                if matches!(opcode, Opcode::Greater|Opcode::GreaterEqual|Opcode::Less|Opcode::LessEqual|Opcode::Add|Opcode::Sub|Opcode::Mul|Opcode::Div) {
                    // TODO(max): Don't guard; string+string is valid too
                    lhs_value = self.push_insn(Opcode::GuardInt, smallvec![lhs_value]);
                    rhs = self.push_insn(Opcode::GuardInt, smallvec![rhs]);
                }
                lhs = LValue::Insn(self.push_insn(opcode, smallvec![lhs_value, rhs]));
            }
            self.lvalue_as_rvalue(env, lhs)
        }
    }
}

mod runtime {
    #[repr(transparent)]
    pub struct Object {
        ptr: usize,
    }

    impl Object {
        #[export_name = "Object_raw"]
        pub extern "C" fn raw(self) -> usize { self.ptr }
    }

    #[repr(C)]
    pub struct HeapObject {
        data: usize,
    }

    #[repr(C)]
    pub struct IntObject {
        obj: HeapObject,
        value: i64,
    }

    pub const kSmallIntTagBits: usize = 1;
    pub const kWordSize: usize = 8;
    pub const kBitsPerByte: usize = 8;
    pub const kBitsPerPointer: usize = kBitsPerByte * kWordSize;
    pub const kBits: usize = kBitsPerPointer - kSmallIntTagBits;
    pub const kSmallIntTagMask: usize = (1 << kSmallIntTagBits) - 1;
    pub const kSmallIntMinValue: isize = -(1isize << (kBits - 1));
    pub const kSmallIntMaxValue: isize = (1isize << (kBits - 1)) - 1;
    pub const kSmallIntTag: usize = 0;         // 0b****0

    extern "C" {
        fn twb_as_heap_object(obj: Object) -> *const HeapObject;
        fn twb_as_int_object(obj: Object) -> *const IntObject;
        fn twb_print(obj: Object) -> std::ffi::c_void;
    }

    #[no_mangle]
    pub extern "C" fn twb_smallint_is_valid(val: isize) -> bool {
        kSmallIntMinValue <= val && val <= kSmallIntMaxValue
    }

    #[no_mangle]
    pub extern "C" fn twb_is_smallint(obj: Object) -> bool {
        (obj.raw() & kSmallIntTagMask) == kSmallIntTag
    }
}

fn main() -> Result<(), hir::ParseError> {
    // let mut lexer = Lexer::from_str("print \"hello, world!\"; 1 + abc <= 3; 4 < 5; 6 == 7; true; false;
    //     var average = (min + max) / 2;");
    let mut lexer = hir::Lexer::from_str("
        var a = 1;
        // if (1) {
        //     a = 2;
        // } else {
        //     a = 3;
        // }
        print a;
        // // (1+2)*3; 4/5; 6 == 7; print 1+8 <= 9; print nil;
        // var x = 1;
        // fun read_global() { return x; }
        // if (1) {
        //     x = 2;
        // } else {
        //     x = 3;
        // }
        // print x;
    ");
    let mut parser = hir::Parser::from_lexer(&mut lexer);
    parser.parse_program()?;
    println!("{}", parser.prog);
    // loop {
    //     let token = lexer.next();
    //     println!("token: {token:?}");
    //     if token.is_err() || token == Ok(Token::Eof) { break; }
    // }
    Ok(())
}

#[cfg(test)]
mod lexer_tests {
    use super::hir::{Lexer, Token};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let lexer = Lexer::from_str(source);
        let actual: Vec<Token> = lexer.collect();
        expect.assert_eq(format!("{actual:?}").as_str());
    }

    fn check_error(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        loop {
            match lexer.next_token() {
                Err(actual) => {
                    expect.assert_eq(format!("{actual:?}").as_str());
                    break;
                }
                Ok(_) => continue,
            }
        }
    }

    #[test]
    fn test_digit() {
        check("1", expect!["[Int(1)]"])
    }

    #[test]
    fn test_digits() {
        check("123", expect!["[Int(123)]"])
    }

    #[test]
    fn test_int_add() {
        check("1+2", expect!["[Int(1), Plus, Int(2)]"])
    }

    #[test]
    fn test_int_mul() {
        check("1*2", expect!["[Int(1), Star, Int(2)]"])
    }

    #[test]
    fn test_int_sub() {
        check("1-2", expect!["[Int(1), Minus, Int(2)]"])
    }

    #[test]
    fn test_int_div() {
        check("1/2", expect!["[Int(1), ForwardSlash, Int(2)]"])
    }

    #[test]
    fn test_var() {
        check("var a = 1;", expect![[r#"[Var, Ident("a"), Equal, Int(1), Semicolon]"#]])
    }

    #[test]
    fn test_ident_char() {
        check("a", expect![[r#"[Ident("a")]"#]])
    }

    #[test]
    fn test_ident_chars() {
        check("abc", expect![[r#"[Ident("abc")]"#]])
    }

    #[test]
    fn test_ident_chars_underscore() {
        check("abc_def", expect![[r#"[Ident("abc_def")]"#]])
    }

    #[test]
    fn test_ident_chars_digits() {
        check("abc123", expect![[r#"[Ident("abc123")]"#]])
    }

    #[test]
    fn test_ident_underscore() {
        check("_", expect![r#"[Ident("_")]"#]);
        check("__", expect![r#"[Ident("__")]"#])
    }

    #[test]
    fn test_if() {
        check("if", expect!["[If]"])
    }

    #[test]
    fn test_while() {
        check("while", expect!["[While]"])
    }

    #[test]
    fn test_comment() {
        check("var a =
               // a comment
               1;",
               expect![[r#"[Var, Ident("a"), Equal, Int(1), Semicolon]"#]])
    }

    #[test]
    fn test_string_lit() {
        check(r#""abc""#, expect![[r#"[Str("abc")]"#]])
    }

    #[test]
    fn test_unterminated_string_lit() {
        check_error("\"abc", expect!["UnterminatedStringLiteral"])
    }

    #[test]
    fn test_comma() {
        check(",", expect!["[Comma]"])
    }

    #[test]
    fn test_dot() {
        check(".", expect!["[Dot]"])
    }
}

#[cfg(test)]
mod type_tests {
    use expect_test::expect;
    use crate::hir_type::*;

    #[test]
    fn test_type_order() {
        assert!(ALL_TYPES.windows(2).all(|w| w[0].1.bits >= w[1].1.bits),
                "ALL_TYPES should be sorted in decreasing order by bit value");
    }

    #[test]
    fn test_int() {
        assert!(TSmallInt.is_subtype(TInt));
        assert!(TLargeInt.is_subtype(TInt));
        assert!(TLargeInt.is_subtype(TAny));
        assert!(TInt.is_subtype(TAny));
        assert!(TEmpty.is_subtype(TAny));
        assert!(TEmpty.is_subtype(TInt));

        assert!(!TInt.is_subtype(TSmallInt));
        assert!(!TInt.is_subtype(TLargeInt));
        assert!(!TAny.is_subtype(TInt));
        assert!(!TAny.is_subtype(TEmpty));
        assert!(!TInt.is_subtype(TEmpty));
    }

    #[test]
    fn test_display_base() {
        expect!["SmallInt"].assert_eq(&TSmallInt.to_string());
        expect!["LargeInt"].assert_eq(&TLargeInt.to_string());
        expect!["Int"].assert_eq(&TInt.to_string());
        expect!["Any"].assert_eq(&TAny.to_string());
        expect!["Empty"].assert_eq(&TEmpty.to_string());
    }

    #[test]
    fn test_display_union() {
        expect!["Str|SmallInt"].assert_eq(&TSmallInt.union(TStr).to_string());
        expect!["CBool|Str|SmallInt"].assert_eq(&TSmallInt.union(TStr).union(TCBool).to_string());
    }
}

#[cfg(test)]
mod parser_tests {
    use super::hir::{Lexer, Parser};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        let mut parser = Parser::from_lexer(&mut lexer);
        parser.parse_program().unwrap();
        let actual = parser.prog;
        expect.assert_eq(format!("{actual}").as_str());
    }

    fn check_error(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        let mut parser = Parser::from_lexer(&mut lexer);
        let result = parser.parse_program();
        expect.assert_eq(format!("{result:?}").as_str())
    }

    #[test]
    fn test_missing_semicolon() {
        check_error("1", expect!["Err(UnexpectedEof)"])
    }

    #[test]
    fn test_toplevel_int_expression_statement() {
        check("1;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:Nil = Const(Nil)
                Return v2
              }
            }
        "#]])
    }

    #[test]
    fn test_equal_equal() {
        check("1==2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:Any = Equal v1, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
        "#]])
    }

    #[test]
    fn test_bang_equal() {
        check("1!=2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:Any = NotEqual v1, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
        "#]])
    }

    // TODO(max): Add tests for prefix not (!)

    #[test]
    fn mul_add() {
        check("1*2+3;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:Int = GuardInt v1
                v4:Int = GuardInt v2
                v5:Any = Mul v3, v4
                v6:SmallInt = Const(Int(3))
                v7:Int = GuardInt v5
                v8:Int = GuardInt v6
                v9:Int = Add v7, v8
                v10:Nil = Const(Nil)
                Return v10
              }
            }
        "#]])
    }

    #[test]
    fn add_mul() {
        check("1+2*3;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:SmallInt = Const(Int(3))
                v4:Int = GuardInt v2
                v5:Int = GuardInt v3
                v6:Any = Mul v4, v5
                v7:Int = GuardInt v1
                v8:Int = GuardInt v6
                v9:Int = Add v7, v8
                v10:Nil = Const(Nil)
                Return v10
              }
            }
        "#]])
    }

    #[test]
    fn test_toplevel_add_expression_statement() {
        check("1+2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:Int = GuardInt v1
                v4:Int = GuardInt v2
                v5:Int = Add v3, v4
                v6:Nil = Const(Nil)
                Return v6
              }
            }
        "#]])
    }

    #[test]
    fn test_or() {
        check("print (1 or 2);", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:CBool = IsTruthy v1
                CondBranch(bb1, bb2) v2
              }
              bb2 {
                v4:SmallInt = Const(Int(2))
                Branch(bb1)
              }
              bb1 {
                v6 = Phi v1, v4
                Print v6
                v8:Nil = Const(Nil)
                Return v8
              }
            }
        "#]])
    }

    #[test]
    fn test_or_or() {
        check("print (1 or 2 or 3);", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:CBool = IsTruthy v1
                CondBranch(bb1, bb2) v2
              }
              bb2 {
                v4:SmallInt = Const(Int(2))
                Branch(bb1)
              }
              bb1 {
                v6 = Phi v1, v4
                v7:CBool = IsTruthy v6
                CondBranch(bb3, bb4) v7
              }
              bb4 {
                v9:SmallInt = Const(Int(3))
                Branch(bb3)
              }
              bb3 {
                v11 = Phi v6, v9
                Print v11
                v13:Nil = Const(Nil)
                Return v13
              }
            }
        "#]])
    }

    #[test]
    fn comment() {
        check("1 +
// a comment
2;", expect![[r#"
    Entry: fn0
    fn0: fun <toplevel>() (entry bb0) {
      bb0 {
        v0:Frame = NewFrame
        v1:SmallInt = Const(Int(1))
        v2:SmallInt = Const(Int(2))
        v3:Int = GuardInt v1
        v4:Int = GuardInt v2
        v5:Int = Add v3, v4
        v6:Nil = Const(Nil)
        Return v6
      }
    }
"#]])
    }

    #[test]
    fn test_print() {
        check("print 1+2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                v2:SmallInt = Const(Int(2))
                v3:Int = GuardInt v1
                v4:Int = GuardInt v2
                v5:Int = Add v3, v4
                Print v5
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_read_unbound_var() {
        check_error("a",
        expect![[r#"Err(UnboundName("a"))"#]])
    }

    #[test]
    fn test_write_unbound_var() {
        check_error("a = 1;",
        expect![[r#"Err(UnboundName("a"))"#]])
    }

    #[test]
    fn test_var() {
        check("
var a = 1;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                Print v3
                v5:Nil = Const(Nil)
                Return v5
              }
            }
        "#]])
    }

    #[test]
    fn test_long_var() {
        check("var aaaabbbbccccdddd = 1;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_assign_non_lvalue() {
        check_error("1 = 2;", expect!["Err(CannotAssignTo(Insn(v1)))"]);
        check_error("var a = 1; (a) = 2;", expect!["Err(CannotAssignTo(Insn(v3)))"]);
        check_error("var a = 1; (a+a) = 2;", expect!["Err(CannotAssignTo(Insn(v7)))"]);
        check_error("class Foo { Foo() { this = 1; } }", expect!["Err(CannotAssignToThis)"]);
        check_error("undefined = 1;", expect![[r#"Err(UnboundName("undefined"))"#]]);
    }

    #[test]
    fn test_var_assign() {
        check("
var a = 1;
a = 2;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@0) v0, v3
                v5:Any = Load(@0) v0
                v6:Any = Load(@0) v0
                Print v6
                v8:Nil = Const(Nil)
                Return v8
              }
            }
        "#]])
    }

    #[test]
    fn test_var_assign_nested() {
        check("
var a = 1;
var b = 2;
var c = 3;
a = b = c;
print a;
print b;
print c;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                v5:SmallInt = Const(Int(3))
                Store(@2) v0, v5
                v7:Any = Load(@2) v0
                Store(@1) v0, v7
                v9:Any = Load(@1) v0
                Store(@0) v0, v9
                v11:Any = Load(@0) v0
                v12:Any = Load(@0) v0
                Print v12
                v14:Any = Load(@1) v0
                Print v14
                v16:Any = Load(@2) v0
                Print v16
                v18:Nil = Const(Nil)
                Return v18
              }
            }
        "#]])
    }

    #[test]
    fn test_var_shadow() {
        check("
var a = 1;
var a = 2;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                v5:Any = Load(@1) v0
                Print v5
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_block() {
        check("{}",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]])
    }

    #[test]
    fn test_if() {
        check("
var a = 1;
if (2) {
    a = 3;
} else {
    a = 4;
}
print a;
",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                v4:CBool = IsTruthy v3
                CondBranch(bb1, bb2) v4
              }
              bb2 {
                v9:SmallInt = Const(Int(4))
                Store(@0) v0, v9
                v11:Any = Load(@0) v0
                Branch(bb3)
              }
              bb1 {
                v6:SmallInt = Const(Int(3))
                Store(@0) v0, v6
                v8:Any = Load(@0) v0
                Branch(bb3)
              }
              bb3 {
                v14:Any = Load(@0) v0
                Print v14
                v16:Nil = Const(Nil)
                Return v16
              }
            }
        "#]])
    }

    #[test]
    fn test_empty_while() {
        check("while (true) {}",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Branch(bb1)
              }
              bb1 {
                v2:Bool = Const(Bool(true))
                v3:CBool = IsTruthy v2
                CondBranch(bb2, bb3) v3
              }
              bb3 {
                v6:Nil = Const(Nil)
                Return v6
              }
              bb2 {
                Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_while_print() {
        check("while (true) { print 1; }",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Branch(bb1)
              }
              bb1 {
                v2:Bool = Const(Bool(true))
                v3:CBool = IsTruthy v2
                CondBranch(bb2, bb3) v3
              }
              bb3 {
                v8:Nil = Const(Nil)
                Return v8
              }
              bb2 {
                v5:SmallInt = Const(Int(1))
                Print v5
                Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_while_complex_cond() {
        check("
            var a = 1;
            while (a < 10) { print a; }",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                Branch(bb1)
              }
              bb1 {
                v4:Any = Load(@0) v0
                v5:SmallInt = Const(Int(10))
                v6:Int = GuardInt v4
                v7:Int = GuardInt v5
                v8:Bool = Less v6, v7
                v9:CBool = IsTruthy v8
                CondBranch(bb2, bb3) v9
              }
              bb3 {
                v14:Nil = Const(Nil)
                Return v14
              }
              bb2 {
                v11:Any = Load(@0) v0
                Print v11
                Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_count_up() {
        check("
            var a = 1;
            while (a < 10) { print a; a = a + 1; }",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                Branch(bb1)
              }
              bb1 {
                v4:Any = Load(@0) v0
                v5:SmallInt = Const(Int(10))
                v6:Int = GuardInt v4
                v7:Int = GuardInt v5
                v8:Bool = Less v6, v7
                v9:CBool = IsTruthy v8
                CondBranch(bb2, bb3) v9
              }
              bb3 {
                v21:Nil = Const(Nil)
                Return v21
              }
              bb2 {
                v11:Any = Load(@0) v0
                Print v11
                v13:Any = Load(@0) v0
                v14:SmallInt = Const(Int(1))
                v15:Int = GuardInt v13
                v16:Int = GuardInt v14
                v17:Int = Add v15, v16
                Store(@0) v0, v17
                v19:Any = Load(@0) v0
                Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_empty_fun() {
        check("fun empty() {}", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun empty() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]])
    }

    // TODO(max): Test that a function can refer to itself

    #[test]
    fn test_scope() {
        check("
var a = 1;
{
    var a = 2;
}
print a;
",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                v5:Any = Load(@0) v0
                Print v5
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_return_nil() {
        check("fun empty() { return nil; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun empty() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_return_param() {
        check("fun f(a) { return a; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_inc() {
        check("fun inc(a) { return a+1; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun inc(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:SmallInt = Const(Int(1))
                v5:Int = GuardInt v3
                v6:Int = GuardInt v4
                v7:Int = Add v5, v6
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_params() {
        check("fun f(a, b) { return a+b; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1, v3) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                Store(@1) v0, v3
                v5:Any = Load(@0) v0
                v6:Any = Load(@1) v0
                v7:Int = GuardInt v5
                v8:Int = GuardInt v6
                v9:Int = Add v7, v8
                Return v9
              }
            }
        "#]])
    }

    #[test]
    fn test_reference_func() {
        check("
            fun f() { }
            f;
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Nil = Const(Nil)
                Return v4
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_call_double_comma() {
        check_error("
            fun f() { }
            f(1,,);
        ", expect!["Err(UnexpectedToken(Comma))"]);
    }

    #[test]
    fn test_call_func_no_args() {
        check("
            fun f() { }
            f();
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Any = Call v3
                v5:Nil = Const(Nil)
                Return v5
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_call_func_one_arg() {
        check("
            fun f() { }
            f(1);
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:SmallInt = Const(Int(1))
                v5:Any = Call v3, v4
                v6:Nil = Const(Nil)
                Return v6
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_call_func_multiple_args() {
        check("
            fun f() { }
            f(1, 2, 3);
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:SmallInt = Const(Int(1))
                v5:SmallInt = Const(Int(2))
                v6:SmallInt = Const(Int(3))
                v7:Any = Call v3, v4, v5, v6
                v8:Nil = Const(Nil)
                Return v8
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_add_call() {
        check("
            fun f() { }
            1+f()+2;
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(1))
                v4:Any = Load(@0) v0
                v5:Any = Call v4
                v6:SmallInt = Const(Int(2))
                v7:Int = GuardInt v5
                v8:Int = GuardInt v6
                v9:Int = Add v7, v8
                v10:Int = GuardInt v3
                v11:Int = GuardInt v9
                v12:Int = Add v10, v11
                v13:Nil = Const(Nil)
                Return v13
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_mul_call() {
        check("
            fun f() { }
            1*f()*2;
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(1))
                v4:Any = Load(@0) v0
                v5:Any = Call v4
                v6:SmallInt = Const(Int(2))
                v7:Int = GuardInt v5
                v8:Int = GuardInt v6
                v9:Any = Mul v7, v8
                v10:Int = GuardInt v3
                v11:Int = GuardInt v9
                v12:Any = Mul v10, v11
                v13:Nil = Const(Nil)
                Return v13
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_call_call() {
        check("
            fun f() { }
            f(1)(2);
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:SmallInt = Const(Int(1))
                v5:Any = Call v3, v4
                v6:SmallInt = Const(Int(2))
                v7:Any = Call v5, v6
                v8:Nil = Const(Nil)
                Return v8
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Nil = Const(Nil)
                Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_load_non_attr() {
        check_error("fun f(a) { return a.1; }", expect!["Err(UnexpectedToken(Int(1)))"])
    }

    #[test]
    fn test_load_attr() {
        check("fun f(a) { return a.b; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Any = GuardInstance v3
                v5:Any = LoadAttr(b) v4
                Return v5
              }
            }
        "#]]);
    }

    #[test]
    fn test_load_attr_nested() {
        check("fun f(a) { return a.b.c; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Any = GuardInstance v3
                v5:Any = LoadAttr(b) v4
                v6:Any = GuardInstance v5
                v7:Any = LoadAttr(c) v6
                Return v7
              }
            }
        "#]]);
    }

    #[test]
    fn test_class() {
        check("class C { }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
        "#]])
    }

    #[test]
    fn test_store_attr() {
        check("fun f(a) { a.b = 1; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:SmallInt = Const(Int(1))
                v5:Any = GuardInstance v3
                v6:Any = StoreAttr(b) v5, v4
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]]);
    }

    #[test]
    fn test_store_attr_nested() {
        check("fun f(a) { a.b.c = 1; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Any = GuardInstance v3
                v5:Any = LoadAttr(b) v4
                v6:SmallInt = Const(Int(1))
                v7:Any = GuardInstance v5
                v8:Any = StoreAttr(c) v7, v6
                v9:Nil = Const(Nil)
                Return v9
              }
            }
        "#]]);
    }

    #[test]
    fn test_load_attr_on_function() {
        check("fun f() { }
            f.a;", expect![[r#"
                Entry: fn0
                fn0: fun <toplevel>() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Any = NewClosure(fn1)
                    Store(@0) v0, v1
                    v3:Any = Load(@0) v0
                    v4:Any = GuardInstance v3
                    v5:Any = LoadAttr(a) v4
                    v6:Nil = Const(Nil)
                    Return v6
                  }
                }
                fn1: fun f() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Nil = Const(Nil)
                    Return v1
                  }
                }
            "#]]);
    }

    #[test]
    fn test_store_attr_on_function() {
        check("fun f() { }
            f.a = 1;", expect![[r#"
                Entry: fn0
                fn0: fun <toplevel>() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Any = NewClosure(fn1)
                    Store(@0) v0, v1
                    v3:Any = Load(@0) v0
                    v4:SmallInt = Const(Int(1))
                    v5:Any = GuardInstance v3
                    v6:Any = StoreAttr(a) v5, v4
                    v7:Nil = Const(Nil)
                    Return v7
                  }
                }
                fn1: fun f() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Nil = Const(Nil)
                    Return v1
                  }
                }
            "#]]);
    }

    #[test]
    fn test_call_method() {
        check("fun f(a) { a.b(); }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Any = Load(@0) v0
                v4:Any = GuardInstance v3
                v5:Any = LoadAttr(b) v4
                v6:Any = Call v5
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_print_class() {
        check("class C { }
        print C;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Any = Load(@0) v0
                Print v4
                v6:Nil = Const(Nil)
                Return v6
              }
            }
        "#]])
    }

    #[test]
    fn test_class_with_empty_method() {
        check("class C {
            empty() { }
        }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
            fn1: fun empty(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_method_with_param() {
        check("class C {
            empty(a) { }
        }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
            fn1: fun empty(v1, v3) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                Store(@1) v0, v3
                v5:Nil = Const(Nil)
                Return v5
              }
            }
        "#]])
    }

    #[test]
    fn test_method_with_params() {
        check("class C {
            empty(a, b) { }
        }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
            fn1: fun empty(v1, v3, v5) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                Store(@1) v0, v3
                Store(@2) v0, v5
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_class_with_empty_methods() {
        check("class C {
            empty0() { }
            empty1() { }
        }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Nil = Const(Nil)
                Return v4
              }
            }
            fn1: fun empty0(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn2: fun empty1(v1) (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_class_inherit() {
        check("class C {}
        class D < C {}", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Any = Load(@0) v0
                v5:Any = GuardClass v4
                v6:Class = NewClass(D, v5)
                Store(@1) v0, v6
                v8:Nil = Const(Nil)
                Return v8
              }
            }
        "#]])
    }

    #[test]
    fn test_class_inherit_self() {
        check_error("class C < C {}", expect![[r#"Err(CannotInheritSelf("C"))"#]])
    }

    // TODO(max): Test referencing class from within method body

    #[test]
    fn test_call_class_no_args() {
        check("class C {}
            C();", expect![[r#"
                Entry: fn0
                fn0: fun <toplevel>() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Class = Const(ObjectClass)
                    v2:Class = NewClass(C, v1)
                    Store(@0) v0, v2
                    v4:Any = Load(@0) v0
                    v5:Any = Call v4
                    v6:Nil = Const(Nil)
                    Return v6
                  }
                }
            "#]])
    }

    #[test]
    fn test_call_class_args() {
        check("class C {}
            C(1, 2);", expect![[r#"
                Entry: fn0
                fn0: fun <toplevel>() (entry bb0) {
                  bb0 {
                    v0:Frame = NewFrame
                    v1:Class = Const(ObjectClass)
                    v2:Class = NewClass(C, v1)
                    Store(@0) v0, v2
                    v4:Any = Load(@0) v0
                    v5:SmallInt = Const(Int(1))
                    v6:SmallInt = Const(Int(2))
                    v7:Any = Call v4, v5, v6
                    v8:Nil = Const(Nil)
                    Return v8
                  }
                }
            "#]])
    }

    #[test]
    fn test_create_class_in_function() {
        check("fun f() { class C { } return C; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Any = NewClosure(fn1)
                Store(@0) v0, v1
                v3:Nil = Const(Nil)
                Return v3
              }
            }
            fn1: fun f() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:Class = Const(ObjectClass)
                v2:Class = NewClass(C, v1)
                Store(@0) v0, v2
                v4:Any = Load(@0) v0
                Return v4
              }
            }
        "#]])
    }

    // #[test]
    // fn test_read_global() {
    //     check("var a = 1; fun empty() { return a; }", expect![[r#"
    //         Entry: fn0
    //         fn0: fun <toplevel> (entry bb0) {
    //           bb0 {
    //             v0 = Const(Nil)
    //             v1 = Return v0
    //           }
    //         }
    //         fn1: fun empty (entry bb0) {
    //           bb0 {
    //             v0 = Const(Nil)
    //             v1 = Return v0
    //           }
    //         }
    //     "#]])
    // }
}

#[cfg(test)]
mod opt_tests {
    use super::hir::{Lexer, Parser};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        let mut parser = Parser::from_lexer(&mut lexer);
        parser.parse_program().unwrap();
        let mut actual = parser.prog;
        for fun in &mut actual.funs {
            fun.unbox_locals();
            fun.infer_types();
            fun.eliminate_dead_code();
        }
        expect.assert_eq(format!("{actual}").as_str());
    }

    #[test]
    fn test_const() {
        check("1; 2; 3;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v4:Nil = Const(Nil)
                Return v4
              }
            }
        "#]])
    }

    #[test]
    fn test_var() {
        check("
var a = 1;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                Print v1
                v5:Nil = Const(Nil)
                Return v5
              }
            }
        "#]])
    }

    #[test]
    fn test_store() {
        check("
var a = 1;
a = 2;
a = 3;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@0) v0, v3
                v6:SmallInt = Const(Int(3))
                Store(@0) v0, v6
                Print v6
                v11:Nil = Const(Nil)
                Return v11
              }
            }
        "#]])
    }

    #[test]
    fn test_store_itself() {
        check("
var a = 1;
a = a;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                Store(@0) v0, v1
                Print v1
                v8:Nil = Const(Nil)
                Return v8
              }
            }
        "#]])
    }

    #[test]
    fn test_store_var() {
        check("
var a = 1;
var b = 2;
a = b;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                Store(@0) v0, v3
                Print v3
                v10:Nil = Const(Nil)
                Return v10
              }
            }
        "#]])
    }

    #[test]
    fn test_var_assign_nested() {
        check("
var a = 1;
var b = 2;
var c = 3;
a = b = c;
print a;
print b;
print c;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                v5:SmallInt = Const(Int(3))
                Store(@2) v0, v5
                Store(@1) v0, v5
                Store(@0) v0, v5
                Print v5
                Print v5
                Print v5
                v18:Nil = Const(Nil)
                Return v18
              }
            }
        "#]])
    }

    #[test]
    fn test_shadow() {
        check("
var a = 1;
var a = 2;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                Store(@1) v0, v3
                Print v3
                v7:Nil = Const(Nil)
                Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_if() {
        check("
var a = 1;
if (2) {
    a = 3;
} else {
    a = 4;
}
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                v3:SmallInt = Const(Int(2))
                v4:CBool = IsTruthy v3
                CondBranch(bb1, bb2) v4
              }
              bb2 {
                v9:SmallInt = Const(Int(4))
                Store(@0) v0, v9
                Branch(bb3)
              }
              bb1 {
                v6:SmallInt = Const(Int(3))
                Store(@0) v0, v6
                Branch(bb3)
              }
              bb3 {
                v18:Any = Phi v6, v9
                Print v18
                v16:Nil = Const(Nil)
                Return v16
              }
            }
        "#]])
    }

    #[test]
    fn test_count_up() {
        check("
            var a = 1;
            while (a < 10) { print a; a = a + 1; }",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel>() (entry bb0) {
              bb0 {
                v0:Frame = NewFrame
                v1:SmallInt = Const(Int(1))
                Store(@0) v0, v1
                Branch(bb1)
              }
              bb1 {
                v23:Any = Phi v1, v17
                v5:SmallInt = Const(Int(10))
                v6:Int = GuardInt v23
                v7:Int = GuardInt v5
                v8:Bool = Less v6, v7
                v9:CBool = IsTruthy v8
                CondBranch(bb2, bb3) v9
              }
              bb3 {
                v21:Nil = Const(Nil)
                Return v21
              }
              bb2 {
                Print v23
                v14:SmallInt = Const(Int(1))
                v15:Int = GuardInt v23
                v16:Int = GuardInt v14
                v17:Int = Add v15, v16
                Store(@0) v0, v17
                Branch(bb1)
              }
            }
        "#]]);
    }
}

#[cfg(test)]
mod runtime_tests {
    use crate::runtime;

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<runtime::HeapObject>(), 8);
    }
}
