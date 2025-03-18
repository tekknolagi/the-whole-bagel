#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
use std::collections::{HashMap, BTreeMap};
use std::collections::VecDeque;
use bit_set::BitSet;

// TODO(max): Set up better testing infrastructure for parse errors


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

    pub fn lookup(&self, id: NameId) -> &str {
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

struct Lexer<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

#[derive(Debug, PartialEq, Clone)]
enum Token {
    Ident(String),
    Print,
    Str(String),
    Int(i64),
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
enum LexError {
    Eof,
    UnexpectedChar(char),
    UnterminatedStringLiteral,
}

const NUMBER_BASE: u32 = 10;
impl<'a> Lexer<'a> {
    fn from_str(source: &'a str) -> Lexer<'a> {
        Lexer { chars: source.chars().peekable() }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        loop {
            match self.chars.next() {
                None => { return Err(LexError::Eof); }
                Some(c) if c.is_whitespace() => { continue; }
                Some(c) if c.is_alphabetic() => { return self.read_ident(c); }
                Some(c) if c.is_digit(NUMBER_BASE) => { return self.read_int(c.to_digit(NUMBER_BASE).unwrap() as i64); }
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
        let mut result: String = "".into();
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

    fn read_int(&mut self, mut result: i64) -> Result<Token, LexError> {
        loop {
            match self.chars.peek() {
                Some(c) if c.is_digit(NUMBER_BASE) => {
                    result *= 10;
                    result += c.to_digit(NUMBER_BASE).unwrap() as i64;
                }
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
}

define_id_type!(":", NameId);
define_id_type!("v", InsnId);
define_id_type!("bb", BlockId);
define_id_type!("fn", FunId);
define_id_type!("@", Offset);

type InsnSet = TypedBitSet<InsnId>;
type BlockSet = TypedBitSet<BlockId>;

#[derive(Debug)]
struct Block {
    insns: Vec<InsnId>,
}

impl Block {
    fn new() -> Block {
        Block { insns: vec![] }
    }
}

#[derive(Debug)]
struct Function {
    name: NameId,
    entry: BlockId,
    insns: Vec<Insn>,
    union_find: Vec<Option<InsnId>>,
    blocks: Vec<Block>,
    num_locals: usize,
}

impl Function {
    fn new(name: NameId) -> Function {
        Function { name, entry: BlockId(0), insns: vec![], union_find: vec![], blocks: vec![Block::new()], num_locals: 0 }
    }

    fn find(&self, insn: InsnId) -> InsnId {
        let mut result = insn;
        loop {
            let it = if result.0 < self.union_find.len() { self.union_find[result.0] } else { None };
            match it {
                Some(insn) => result = insn,
                None => return result,
            }
        }
    }

    fn make_equal_to(&mut self, left: InsnId, right: InsnId) {
        let found = self.find(left);
        if found.0 >= self.union_find.len() {
            self.union_find.resize(found.0+1, None);
        }
        self.union_find[found.0] = Some(right);
    }

    fn new_block(&mut self) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::new());
        result
    }

    fn new_insn(&mut self, insn: Insn) -> InsnId {
        let result = InsnId(self.insns.len());
        self.insns.push(insn);
        result
    }

    fn is_terminated(&self, block: BlockId) -> bool {
        match self.blocks[block.0].insns.last().map(|insn| &self.insns[insn.0].opcode) {
            Some(Opcode::Return | Opcode::CondBranch(..) | Opcode::Branch(..)) => true,
            _ => false,
        }
    }

    fn succs(&self, block: BlockId) -> Vec<BlockId> {
        match self.blocks[block.0].insns.last().map(|insn| &self.insns[insn.0].opcode) {
            Some(Opcode::Return) => vec![],
            Some(Opcode::CondBranch(iftrue, iffalse)) => vec![*iftrue, *iffalse],
            Some(Opcode::Branch(target)) => vec![*target],
            _ => vec![],
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

    fn unbox_locals(&mut self) {
        fn join(left: &Vec<InsnSet>, right: &Vec<InsnSet>) -> Vec<InsnSet> {
            assert_eq!(left.len(), right.len());
            let mut result = vec![InsnSet::new(); left.len()];
            for idx in 0..left.len() {
                result[idx] = left[idx].union(&right[idx]);
            }
            result
        }
        let empty_state = vec![InsnSet::new(); self.num_locals];
        let mut block_entry = vec![empty_state.clone(); self.blocks.len()];
        let mut replacements: BTreeMap<InsnId, InsnSet> = BTreeMap::new();
        let mut last_pass = false;
        loop {
            let mut changed = false;
            for block_id in self.rpo() {
                let mut env: Vec<_> = block_entry[block_id.0].clone();
                for insn_id in &self.blocks[block_id.0].insns {
                    let Insn { opcode, operands } = &self.insns[insn_id.0];
                    match opcode {
                        Opcode::Store(offset) => {
                            env[offset.0] = InsnSet::one(self.find(operands[1]));
                        }
                        Opcode::Load(offset) if last_pass => {
                            replacements.insert(*insn_id, env[offset.0].clone());
                        }
                        _ => {}
                    }
                }
                for succ in self.succs(block_id) {
                    let new = join(&block_entry[succ.0], &env);
                    if block_entry[succ.0] != new {
                        block_entry[succ.0] = new;
                        changed = true;
                    }
                }
            }
            if last_pass {
                break;
            }
            if !changed {
                last_pass = true;
            }
        }
        // TODO(max): Hash-cons phi
        for (insn_id, mut operands) in replacements {
            match operands.len() {
                0 => panic!("Should have at least one value"),
                1 => self.make_equal_to(insn_id, operands.get_single()),
                _ => {
                    let phi = self.new_insn(Insn { opcode: Opcode::Phi, operands: operands.as_vec() });
                    self.make_equal_to(insn_id, phi);
                }
            }
        }
    }

    fn is_critical(&self, insn: InsnId) -> bool {
        match &self.insns[insn.0].opcode {
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
            Opcode::GuardInt => true,
            Opcode::NewClass(_) => false,
            Opcode::This => false,
            Opcode::NewClosure(_) => false,
            Opcode::Call(_) => true,
        }
    }

    fn eliminate_dead_code(&mut self) {
        let mut worklist = VecDeque::new();
        for block_id in self.rpo() {
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
        for block_id in self.rpo() {
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
        writeln!(f, "fun {fun_name} (entry {fun_entry}) {{")?;
        let mut seen = InsnSet::new();
        for block_id in fun.rpo() {
            writeln!(f, "  {block_id} {{")?;
            for insn_id in &fun.blocks[block_id.0].insns {
                let insn_id = fun.find(*insn_id);
                if seen.contains(insn_id) { continue; }
                seen.insert(insn_id);
                let Insn { opcode, operands } = &fun.insns[insn_id.0];
                match opcode {
                    Opcode::NewClass(ClassDef { name, .. }) => {
                        let class_name = self.program.interner.lookup(*name);
                        write!(f, "    {insn_id} = NewClass({class_name})")
                    }
                    _ => write!(f, "    {insn_id} = {:?}", opcode),
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
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

#[derive(Debug)]
struct ClassDef {
    name: NameId,
    methods: Vec<FunId>,
}

#[derive(Debug)]
enum Opcode {
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
    Load(Offset),
    Store(Offset),
    GuardInt,
    This,
    Call(InsnId),
}

#[derive(Debug)]
struct Insn {
    opcode: Opcode,
    // TODO(max): Use smallvec
    operands: Vec<InsnId>,
}

struct Program {
    entry: FunId,
    funs: Vec<Function>,
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
        let name = result.intern_str("<toplevel>");
        result.funs[0].name = name;
        result
    }

    fn intern(&mut self, name: String) -> NameId {
        self.interner.intern(&name)
    }

    fn intern_str(&mut self, name: &str) -> NameId {
        self.interner.intern(name)
    }

    fn push_fun(&mut self, name: NameId) -> FunId {
        let fun = FunId(self.funs.len());
        self.funs.push(Function::new(name));
        fun
    }

    fn push_insn(&mut self, fun: FunId, block: BlockId, opcode: Opcode, operands: Vec<InsnId>) -> InsnId {
        // TODO(max): Catch double terminators
        let result = self.funs[fun.0].new_insn(Insn { opcode, operands });
        self.funs[fun.0].blocks[block.0].insns.push(result);
        result
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
    bindings: HashMap<NameId, Offset>,
    parent: Option<&'a Env<'a>>,
}

impl<'a> Env<'a> {
    fn new(fun: FunId) -> Env<'a> {
        Env { fun, bindings: HashMap::new(), parent: None }
    }

    fn from_parent(fun: FunId, parent: &'a Env<'a>) -> Env<'a> {
        Env { fun, bindings: HashMap::new(), parent: Some(parent) }
    }

    fn lookup(&self, name: NameId) -> Option<Offset> {
        self.bindings.get(&name).copied()
    }

    fn is_defined(&self, name: NameId) -> bool {
        return self.bindings.get(&name).is_some();
    }

    fn define(&mut self, name: NameId) -> Offset {
        let offset = Offset(self.bindings.len());
        self.bindings.insert(name.clone(), offset);
        offset
    }
}

struct Context {
    fun: FunId,
    block: BlockId,
    frame: InsnId,
}

struct Parser<'a> {
    tokens: std::iter::Peekable<&'a mut Lexer<'a>>,
    prog: Program,
    context_stack: Vec<Context>,
}

#[derive(Debug, PartialEq)]
enum ParseError {
    UnexpectedToken(Token),
    UnexpectedEof,
    UnboundName(String),
    VariableShadows(String),
}

impl Parser<'_> {
    fn from_lexer<'a>(lexer: &'a mut Lexer<'a>) -> Parser<'a> {
        Parser { tokens: lexer.peekable(), prog: Program::new(), context_stack: vec![] }
    }

    fn enter_fun(&mut self, fun: FunId) {
        let block = self.prog.funs[fun.0].entry;
        let frame = self.prog.push_insn(fun, block, Opcode::NewFrame, vec![]);
        self.context_stack.push(Context { fun, block, frame });
        self.enter_block(block);
    }

    fn leave_fun(&mut self) {
        if !self.prog.funs[self.fun().0].is_terminated(self.block()) {
            let nil = self.push_insn(Opcode::Const(Value::Nil), vec![]);
            self.push_insn(Opcode::Return, vec![nil]);
        }
        if let Some(Context { fun, block, frame }) = self.context_stack.pop() {
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
            Some(Token::Ident(name)) => Ok(self.prog.intern(name)),
            None => Err(ParseError::UnexpectedEof),
            Some(actual) => Err(ParseError::UnexpectedToken(actual)),
        }
    }

    fn push_insn(&mut self, opcode: Opcode, operands: Vec<InsnId>) -> InsnId {
        self.prog.push_insn(self.fun(), self.block(), opcode, operands)
    }

    fn parse_program(&mut self) -> Result<(), ParseError> {
        self.enter_fun(self.prog.entry);
        let mut env = Env::new(self.prog.entry);
        while let Some(token) = self.tokens.peek() {
            self.parse_toplevel(&mut env)?;
        }
        self.leave_fun();
        Ok(())
    }

    fn parse_class(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
        self.expect(Token::Class)?;
        let name = self.expect_ident()?;
        if let Some(Token::Less) = self.tokens.peek() {
            todo!("inheritance");
        }
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
        let class = self.push_insn(Opcode::NewClass(ClassDef { name, methods }), vec![]);
        let offset = env.define(name);
        self.write_local(offset, class);
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
        let this_name = self.prog.intern_str("this");
        let offset = func_env.define(this_name);
        let this = self.push_insn(Opcode::This, vec![]);
        self.write_local(offset, this);
        loop {
            match self.tokens.peek() {
                Some(Token::Ident(name)) => {
                    let name = self.prog.intern_str(name);
                    let param = self.push_insn(Opcode::Param(idx), vec![]);
                    let offset = func_env.define(name);
                    self.write_local(offset, param);
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
            Some(Token::Class) => self.parse_class(&mut env),
            _ => self.parse_statement(&mut env),
        }
    }

    fn parse_statement(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
        match self.tokens.peek() {
            Some(Token::Print) => {
                self.tokens.next();
                let expr = self.parse_expression(&mut env)?;
                self.push_insn(Opcode::Print, vec![expr]);
                Ok(())
            }
            Some(Token::Return) => {
                self.tokens.next();
                let expr = self.parse_expression(&mut env)?;
                self.push_insn(Opcode::Return, vec![expr]);
                Ok(())
            }
            Some(Token::Fun) => {
                self.tokens.next();
                return self.parse_function(&mut env);  // no semicolon
            }
            Some(Token::Var) => {
                self.tokens.next();
                let name = self.expect_ident()?;
                self.expect(Token::Equal)?;
                let value = self.parse_expression(&mut env)?;
                let offset = env.define(name);
                self.write_local(offset, value);
                Ok(())
            }
            Some(Token::If) => {
                self.tokens.next();
                self.expect(Token::LParen)?;
                let cond = self.parse_expression(&mut env)?;
                self.expect(Token::RParen)?;
                let iftrue_block = self.new_block();
                let iffalse_block = self.new_block();
                self.push_insn(Opcode::CondBranch(iftrue_block, iffalse_block), vec![cond]);

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
                    self.push_insn(Opcode::Branch(join_block), vec![]);

                    self.enter_block(join_block);
                    self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(join_block), vec![]);
                } else {
                    self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(iffalse_block), vec![]);
                }
                return Ok(());  // no semicolon
            }
            Some(Token::While) => {
                self.tokens.next();
                self.expect(Token::LParen)?;
                let header_block = self.new_block();
                self.push_insn(Opcode::Branch(header_block), vec![]);
                self.enter_block(header_block);
                let cond = self.parse_expression(&mut env)?;
                self.expect(Token::RParen)?;
                let body_block = self.new_block();
                let after_block = self.new_block();
                self.push_insn(Opcode::CondBranch(body_block, after_block), vec![cond]);
                self.enter_block(body_block);
                let mut body_env = env.clone();
                self.parse_statement(&mut body_env)?;
                self.push_insn(Opcode::Branch(header_block), vec![]);
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
            Some(token) => { self.parse_expression(&mut env)?; Ok(()) },
            None => { Err(ParseError::UnexpectedEof) }
        }?;
        self.expect(Token::Semicolon)
    }

    fn write_local(&mut self, offset: Offset, value: InsnId) {
        let fun_id = self.fun();
        let num_locals = &mut self.prog.funs[fun_id.0].num_locals;
        *num_locals = std::cmp::max(*num_locals, offset.0) + 1;
        self.push_insn(Opcode::Store(offset), vec![self.frame(), value]);
    }

    fn read_local(&mut self, offset: Offset) -> InsnId {
        let fun_id = self.fun();
        let num_locals = &mut self.prog.funs[fun_id.0].num_locals;
        *num_locals = std::cmp::max(*num_locals, offset.0) + 1;
        self.push_insn(Opcode::Load(offset), vec![self.frame()])
    }

    fn parse_function(&mut self, mut env: &mut Env) -> Result<(), ParseError> {
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
                    let name = self.prog.intern_str(name);
                    let param = self.push_insn(Opcode::Param(idx), vec![]);
                    let offset = func_env.define(name);
                    self.write_local(offset, param);
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
        let closure = self.push_insn(Opcode::NewClosure(fun), vec![]);
        let offset = env.define(name);
        self.write_local(offset, closure);
        Ok(())
    }

    fn parse_args(&mut self, mut env: &mut Env) -> Result<Vec<InsnId>, ParseError> {
        // TODO(max): Come up with a better idiom to parse comma-separated lists. This is wack.
        // TODO(max): Disallow f(x,)
        let mut result = vec![];
        loop {
            match self.tokens.peek() {
                Some(Token::RParen) => break,
                Some(Token::Comma) => return Err(ParseError::UnexpectedToken(Token::Comma)),
                None => return Err(ParseError::UnexpectedEof),
                _ => result.push(self.parse_expression(&mut env)?),
            }
            match self.tokens.peek() {
                Some(Token::Comma) => { self.tokens.next(); }
                _ => break,
            }
        }
        Ok(result)
    }

    fn parse_expression(&mut self, mut env: &mut Env) -> Result<InsnId, ParseError> {
        self.parse_(env, 0)
    }

    fn parse_(&mut self, mut env: &mut Env, prec: u32) -> Result<InsnId, ParseError> {
        let mut lhs = match self.tokens.peek() {
            None => Err(ParseError::UnexpectedEof),
            Some(Token::Nil) => {
                self.tokens.next();
                Ok(self.push_insn(Opcode::Const(Value::Nil), vec![]))
            }
            Some(Token::Bool(value)) => {
                let value = value.clone();
                self.tokens.next();
                Ok(self.push_insn(Opcode::Const(Value::Bool(value)), vec![]))
            }
            Some(Token::Int(value)) => {
                let value = value.clone();
                self.tokens.next();
                Ok(self.push_insn(Opcode::Const(Value::Int(value)), vec![]))
            }
            Some(Token::Str(value)) => {
                let value = value.clone();
                self.tokens.next();
                Ok(self.push_insn(Opcode::Const(Value::Str(value)), vec![]))
            }
            Some(Token::Ident(name)) => {
                let name_str = name.clone();
                let name = self.prog.intern_str(name);
                self.tokens.next();
                if self.tokens.peek() == Some(&Token::Equal) {
                    // Assignment
                    self.tokens.next();
                    // TODO(max): Operator precedence ...?
                    let value = self.parse_expression(&mut env)?;
                    match env.lookup(name) {
                        Some(offset) => {
                            self.write_local(offset, value);
                            Ok(value)
                        }
                        None => Err(ParseError::UnboundName(name_str)),
                    }
                } else {
                    match env.lookup(name) {
                        Some(offset) => Ok(self.read_local(offset)),
                        None => Err(ParseError::UnboundName(name_str)),
                    }
                }
            }
            Some(Token::LParen) => {
                self.tokens.next();
                let result = self.parse_(&mut env, 0)?;
                self.expect(Token::RParen)?;
                Ok(result)
            }
            Some(token) => Err(ParseError::UnexpectedToken(token.clone())),
        }?;
        while let Some(token) = self.tokens.peek() {
            let (assoc, op_prec) = match token {
                Token::And => (Assoc::Left, 0),
                Token::Or => (Assoc::Left, 0),
                Token::EqualEqual => (Assoc::Left, 1),
                Token::BangEqual => (Assoc::Left, 1),
                Token::Greater => (Assoc::Left, 2),
                Token::GreaterEqual => (Assoc::Left, 2),
                Token::Less => (Assoc::Left, 2),
                Token::LessEqual => (Assoc::Left, 2),
                Token::Plus => (Assoc::Any, 3),
                Token::Minus => (Assoc::Left, 3),
                Token::Star => (Assoc::Any, 4),
                Token::ForwardSlash => (Assoc::Left, 4),
                Token::LParen => (Assoc::Any, 5),
                _ => break,
            };
            let token = token.clone();
            if op_prec < prec { return Ok(lhs); }
            self.tokens.next();
            let next_prec = if assoc == Assoc::Left { op_prec + 1 } else { op_prec };
            if token == Token::And {
                todo!("ssa from `and' keyword")
            } else if token == Token::Or {
                let iftrue_block = self.new_block();
                let iffalse_block = self.new_block();
                self.push_insn(Opcode::CondBranch(iftrue_block, iffalse_block), vec![lhs.clone()]);
                self.enter_block(iffalse_block);
                let rhs = self.parse_(&mut env, next_prec)?;
                self.push_insn(Opcode::Branch(iftrue_block), vec![]);
                self.enter_block(iftrue_block);
                lhs = self.push_insn(Opcode::Phi, vec![lhs, rhs])
            } else if token == Token::LParen {
                // Function call
                let operands = self.parse_args(&mut env)?;
                self.expect(Token::RParen)?;
                lhs = self.push_insn(Opcode::Call(lhs), operands)
            } else {
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
                    lhs = self.push_insn(Opcode::GuardInt, vec![lhs]);
                    rhs = self.push_insn(Opcode::GuardInt, vec![rhs]);
                }
                lhs = self.push_insn(opcode, vec![lhs, rhs]);
            }
        }
        Ok(lhs)
    }
}

fn main() -> Result<(), ParseError> {
    // let mut lexer = Lexer::from_str("print \"hello, world!\"; 1 + abc <= 3; 4 < 5; 6 == 7; true; false;
    //     var average = (min + max) / 2;");
    let mut lexer = Lexer::from_str("
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
    let mut parser = Parser::from_lexer(&mut lexer);
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
    use super::{Lexer, Token};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
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

    // #[test]
    // fn test_ident_underscore() {
    //     check("_", expect!["[]"])
    // }

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
mod parser_tests {
    use super::{Lexer, Parser};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        let mut parser = Parser::from_lexer(&mut lexer);
        parser.parse_program().unwrap();
        let actual = parser.prog;
        expect.assert_eq(format!("{actual}").as_str());
    }

    #[test]
    fn test_toplevel_int_expression_statement() {
        check("1;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Const(Nil)
                v3 = Return v2
              }
            }
        "#]])
    }

    #[test]
    fn mul_add() {
        check("1*2+3;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Const(Int(2))
                v3 = GuardInt v1
                v4 = GuardInt v2
                v5 = Mul v3, v4
                v6 = Const(Int(3))
                v7 = GuardInt v5
                v8 = GuardInt v6
                v9 = Add v7, v8
                v10 = Const(Nil)
                v11 = Return v10
              }
            }
        "#]])
    }

    #[test]
    fn add_mul() {
        check("1+2*3;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Const(Int(2))
                v3 = Const(Int(3))
                v4 = GuardInt v2
                v5 = GuardInt v3
                v6 = Mul v4, v5
                v7 = GuardInt v1
                v8 = GuardInt v6
                v9 = Add v7, v8
                v10 = Const(Nil)
                v11 = Return v10
              }
            }
        "#]])
    }

    #[test]
    fn test_toplevel_add_expression_statement() {
        check("1+2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Const(Int(2))
                v3 = GuardInt v1
                v4 = GuardInt v2
                v5 = Add v3, v4
                v6 = Const(Nil)
                v7 = Return v6
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
    fn0: fun <toplevel> (entry bb0) {
      bb0 {
        v0 = NewFrame
        v1 = Const(Int(1))
        v2 = Const(Int(2))
        v3 = GuardInt v1
        v4 = GuardInt v2
        v5 = Add v3, v4
        v6 = Const(Nil)
        v7 = Return v6
      }
    }
"#]])
    }

    #[test]
    fn test_print() {
        check("print 1+2;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Const(Int(2))
                v3 = GuardInt v1
                v4 = GuardInt v2
                v5 = Add v3, v4
                v6 = Print v5
                v7 = Const(Nil)
                v8 = Return v7
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Print v3
                v5 = Const(Nil)
                v6 = Return v5
              }
            }
        "#]])
    }

    #[test]
    fn test_var_assign() {
        check("
var a = 1;
a = 2;
print a;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@0) v0, v3
                v5 = Load(@0) v0
                v6 = Print v5
                v7 = Const(Nil)
                v8 = Return v7
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@1) v0, v3
                v5 = Load(@1) v0
                v6 = Print v5
                v7 = Const(Nil)
                v8 = Return v7
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = CondBranch(bb1, bb2) v3
              }
              bb2 {
                v7 = Const(Int(4))
                v8 = Store(@0) v0, v7
                v9 = Branch(bb3)
              }
              bb1 {
                v5 = Const(Int(3))
                v6 = Store(@0) v0, v5
                v10 = Branch(bb3)
              }
              bb3 {
                v11 = Load(@0) v0
                v12 = Print v11
                v13 = Const(Nil)
                v14 = Return v13
              }
            }
        "#]])
    }

    #[test]
    fn test_empty_while() {
        check("while (true) {}",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Branch(bb1)
              }
              bb1 {
                v2 = Const(Bool(true))
                v3 = CondBranch(bb2, bb3) v2
              }
              bb3 {
                v5 = Const(Nil)
                v6 = Return v5
              }
              bb2 {
                v4 = Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_while_print() {
        check("while (true) { print 1; }",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Branch(bb1)
              }
              bb1 {
                v2 = Const(Bool(true))
                v3 = CondBranch(bb2, bb3) v2
              }
              bb3 {
                v7 = Const(Nil)
                v8 = Return v7
              }
              bb2 {
                v4 = Const(Int(1))
                v5 = Print v4
                v6 = Branch(bb1)
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Branch(bb1)
              }
              bb1 {
                v4 = Load(@0) v0
                v5 = Const(Int(10))
                v6 = GuardInt v4
                v7 = GuardInt v5
                v8 = Less v6, v7
                v9 = CondBranch(bb2, bb3) v8
              }
              bb3 {
                v13 = Const(Nil)
                v14 = Return v13
              }
              bb2 {
                v10 = Load(@0) v0
                v11 = Print v10
                v12 = Branch(bb1)
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Branch(bb1)
              }
              bb1 {
                v4 = Load(@0) v0
                v5 = Const(Int(10))
                v6 = GuardInt v4
                v7 = GuardInt v5
                v8 = Less v6, v7
                v9 = CondBranch(bb2, bb3) v8
              }
              bb3 {
                v19 = Const(Nil)
                v20 = Return v19
              }
              bb2 {
                v10 = Load(@0) v0
                v11 = Print v10
                v12 = Load(@0) v0
                v13 = Const(Int(1))
                v14 = GuardInt v12
                v15 = GuardInt v13
                v16 = Add v14, v15
                v17 = Store(@0) v0, v16
                v18 = Branch(bb1)
              }
            }
        "#]]);
    }

    #[test]
    fn test_empty_fun() {
        check("fun empty() {}", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun empty (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
              }
            }
        "#]])
    }

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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@1) v0, v3
                v5 = Load(@0) v0
                v6 = Print v5
                v7 = Const(Nil)
                v8 = Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_return_nil() {
        check("fun empty() { return nil; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun empty (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_return_param() {
        check("fun f(a) { return a; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Param(0)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_inc() {
        check("fun inc(a) { return a+1; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun inc (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Param(0)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Const(Int(1))
                v5 = GuardInt v3
                v6 = GuardInt v4
                v7 = Add v5, v6
                v8 = Return v7
              }
            }
        "#]])
    }

    #[test]
    fn test_fun_params() {
        check("fun f(a, b) { return a+b; }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Param(0)
                v2 = Store(@0) v0, v1
                v3 = Param(1)
                v4 = Store(@1) v0, v3
                v5 = Load(@0) v0
                v6 = Load(@1) v0
                v7 = GuardInt v5
                v8 = GuardInt v6
                v9 = Add v7, v8
                v10 = Return v9
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Const(Nil)
                v5 = Return v4
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_call_func_no_args() {
        check("
            fun f() { }
            f();
        ", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Call(v3)
                v5 = Const(Nil)
                v6 = Return v5
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Const(Int(1))
                v5 = Call(v3) v4
                v6 = Const(Nil)
                v7 = Return v6
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Const(Int(1))
                v5 = Const(Int(2))
                v6 = Const(Int(3))
                v7 = Call(v3) v4, v5, v6
                v8 = Const(Nil)
                v9 = Return v8
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Int(1))
                v4 = Load(@0) v0
                v5 = Call(v4)
                v6 = Const(Int(2))
                v7 = GuardInt v5
                v8 = GuardInt v6
                v9 = Add v7, v8
                v10 = GuardInt v3
                v11 = GuardInt v9
                v12 = Add v10, v11
                v13 = Const(Nil)
                v14 = Return v13
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Const(Int(1))
                v4 = Load(@0) v0
                v5 = Call(v4)
                v6 = Const(Int(2))
                v7 = GuardInt v5
                v8 = GuardInt v6
                v9 = Mul v7, v8
                v10 = GuardInt v3
                v11 = GuardInt v9
                v12 = Mul v10, v11
                v13 = Const(Nil)
                v14 = Return v13
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClosure(fn1)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Const(Int(1))
                v5 = Call(v3) v4
                v6 = Const(Int(2))
                v7 = Call(v5) v6
                v8 = Const(Nil)
                v9 = Return v8
              }
            }
            fn1: fun f (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Nil)
                v2 = Return v1
              }
            }
        "#]]);
    }

    #[test]
    fn test_class() {
        check("class C { }", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClass(C)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
        "#]])
    }

    #[test]
    fn test_print_class() {
        check("class C { }
        print C;", expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClass(C)
                v2 = Store(@0) v0, v1
                v3 = Load(@0) v0
                v4 = Print v3
                v5 = Const(Nil)
                v6 = Return v5
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClass(C)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun empty (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = This
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = NewClass(C)
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn1: fun empty0 (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = This
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
              }
            }
            fn2: fun empty1 (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = This
                v2 = Store(@0) v0, v1
                v3 = Const(Nil)
                v4 = Return v3
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
    use super::{Lexer, Parser};
    use expect_test::{expect, Expect};

    fn check(source: &str, expect: Expect) {
        let mut lexer = Lexer::from_str(source);
        let mut parser = Parser::from_lexer(&mut lexer);
        parser.parse_program().unwrap();
        let mut actual = parser.prog;
        for fun in &mut actual.funs {
            fun.unbox_locals();
            fun.eliminate_dead_code();
        }
        expect.assert_eq(format!("{actual}").as_str());
    }

    #[test]
    fn test_const() {
        check("1; 2; 3;",
        expect![[r#"
            Entry: fn0
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v4 = Const(Nil)
                v5 = Return v4
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v4 = Print v1
                v5 = Const(Nil)
                v6 = Return v5
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@0) v0, v3
                v5 = Const(Int(3))
                v6 = Store(@0) v0, v5
                v8 = Print v5
                v9 = Const(Nil)
                v10 = Return v9
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v4 = Store(@0) v0, v1
                v6 = Print v1
                v7 = Const(Nil)
                v8 = Return v7
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@1) v0, v3
                v6 = Store(@0) v0, v3
                v8 = Print v3
                v9 = Const(Nil)
                v10 = Return v9
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = Store(@1) v0, v3
                v6 = Print v3
                v7 = Const(Nil)
                v8 = Return v7
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Const(Int(2))
                v4 = CondBranch(bb1, bb2) v3
              }
              bb2 {
                v7 = Const(Int(4))
                v8 = Store(@0) v0, v7
                v9 = Branch(bb3)
              }
              bb1 {
                v5 = Const(Int(3))
                v6 = Store(@0) v0, v5
                v10 = Branch(bb3)
              }
              bb3 {
                v15 = Phi v5, v7
                v12 = Print v15
                v13 = Const(Nil)
                v14 = Return v13
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
            fn0: fun <toplevel> (entry bb0) {
              bb0 {
                v0 = NewFrame
                v1 = Const(Int(1))
                v2 = Store(@0) v0, v1
                v3 = Branch(bb1)
              }
              bb1 {
                v21 = Phi v1, v16
                v5 = Const(Int(10))
                v6 = GuardInt v21
                v7 = GuardInt v5
                v8 = Less v6, v7
                v9 = CondBranch(bb2, bb3) v8
              }
              bb3 {
                v19 = Const(Nil)
                v20 = Return v19
              }
              bb2 {
                v22 = Phi v1, v16
                v11 = Print v22
                v23 = Phi v1, v16
                v13 = Const(Int(1))
                v14 = GuardInt v23
                v15 = GuardInt v13
                v16 = Add v14, v15
                v17 = Store(@0) v0, v16
                v18 = Branch(bb1)
              }
            }
        "#]]);
    }
}
