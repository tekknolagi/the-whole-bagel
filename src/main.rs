use std::collections::{HashMap, HashSet};

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
                Some(c) if c.is_alphabetic() || *c == '_' => result.push(*c),
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

#[derive(PartialEq, Copy, Clone)]
struct InsnId(usize);
#[derive(PartialEq, Copy, Clone)]
struct BlockId(usize);
#[derive(PartialEq, Copy, Clone)]
struct FunId(usize);

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "v{}", self.0)
    }
}

impl std::fmt::Debug for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "v{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "bb{}", self.0)
    }
}

impl std::fmt::Debug for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "bb{}", self.0)
    }
}

impl std::fmt::Display for FunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "fn{}", self.0)
    }
}

impl std::fmt::Debug for FunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "fn{}", self.0)
    }
}

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
    name: String,
    entry: BlockId,
    insns: Vec<Insn>,
    union_find: Vec<Option<InsnId>>,
    blocks: Vec<Block>,
}

impl Function {
    fn new(name: String) -> Function {
        Function { name, entry: BlockId(0), insns: vec![], union_find: vec![], blocks: vec![Block::new()] }
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
            self.union_find.resize(found.0, None);
        }
        self.union_find[found.0] = Some(right);
    }

    fn new_block(&mut self) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::new());
        result
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "fun {} (entry {}) {{", self.name, self.entry)?;
        for (idx, block) in self.blocks.iter().enumerate() {
            let block_id = BlockId(idx);
            writeln!(f, "  {block_id} {{")?;
            for insn_id in &self.blocks[block_id.0].insns {
                writeln!(f, "    {insn_id} = {:?}", self.insns[insn_id.0])?;
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
enum Opcode {
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
    ClosureRef(usize),
    ClosureSet(usize),
    Branch(BlockId),
    CondBranch(BlockId, BlockId),
    Phi,
}

#[derive(Clone, PartialEq)]
enum Opnd {
    Insn(InsnId),
    Const(Value),
}

impl std::fmt::Debug for Opnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Opnd::Insn(insn_id) => write!(f, "{insn_id}"),
            Opnd::Const(val) => write!(f, "{val:?}"),
        }
    }
}

#[derive(Debug)]
struct Insn {
    opcode: Opcode,
    operands: Vec<Opnd>,
}

#[derive(Debug)]
struct Program {
    entry: FunId,
    funs: Vec<Function>,
}

#[derive(PartialEq)]
enum Assoc {
    Any,
    Left,
    Right
}

impl Program {
    fn new() -> Program {
        let main = Function::new("<toplevel>".into());
        Program { entry: FunId(0), funs: vec![main] }
    }

    fn push_fun(&mut self, name: String) -> FunId {
        let fun = FunId(self.funs.len());
        self.funs.push(Function::new(name));
        fun
    }

    fn push_insn(&mut self, fun: FunId, block: BlockId, opcode: Opcode, operands: Vec<Opnd>) -> InsnId {
        // TODO(max): Catch double terminators
        let result = InsnId(self.funs[fun.0].insns.len());
        self.funs[fun.0].insns.push(Insn { opcode, operands });
        self.funs[fun.0].blocks[block.0].insns.push(result);
        result
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Entry: {}", self.entry)?;
        for (idx, fun) in self.funs.iter().enumerate() {
            let fun_id = FunId(idx);
            write!(f, "{fun_id}: {fun}")?;
        }
        Ok(())
    }
}

enum VarKind {
    Closure,
    Local,
}

#[derive(Clone)]
struct Env<'a> {
    fun: FunId,
    bindings: HashMap<String, Opnd>,
    parent: Option<&'a Env<'a>>,
}

impl<'a> Env<'a> {
    fn new(fun: FunId) -> Env<'a> {
        Env { fun, bindings: HashMap::new(), parent: None }
    }

    fn from_parent(fun: FunId, parent: &'a Env<'a>) -> Env<'a> {
        Env { fun, bindings: HashMap::new(), parent: Some(parent) }
    }

    fn lookup(&self, name: &String) -> Option<(VarKind, Opnd)> {
        self.lookup_(VarKind::Local, name)
    }

    fn lookup_(&self, kind: VarKind, name: &String) -> Option<(VarKind, Opnd)> {
        match self.bindings.get(name) {
            None if self.parent.is_none() => None,
            None => self.parent.as_ref().unwrap().lookup_(VarKind::Closure, name),
            Some(value) => Some((kind, value.clone())),
        }
    }

    fn define(&mut self, name: &String, value: Opnd) {
        assert!(self.bindings.get(name).is_none());
        self.bindings.insert(name.clone(), value);
    }

    fn set(&mut self, name: &String, value: Opnd) -> Option<VarKind> {
        use std::collections::hash_map::Entry;
        match (self.bindings.entry(name.clone()), &mut self.parent) {
            (Entry::Vacant(entry), None) => None,
            (Entry::Vacant(entry), Some(_)) => Some(VarKind::Closure),
            (Entry::Occupied(mut entry), _) => { entry.insert(value); Some(VarKind::Local) }
        }
    }
}

struct Context {
    fun: FunId,
    block: BlockId,
}

struct Parser<'a> {
    tokens: std::iter::Peekable<&'a mut Lexer<'a>>,
    prog: Program,
    context_stack: Vec<Context>,
}

#[derive(Debug, PartialEq)]
enum ParseError {
    UnexpectedToken(Token),
    UnexpectedError,
    UnboundName(String),
}

impl Parser<'_> {
    fn from_lexer<'a>(lexer: &'a mut Lexer<'a>) -> Parser<'a> {
        Parser { tokens: lexer.peekable(), prog: Program::new(), context_stack: vec![] }
    }

    fn enter_fun(&mut self, fun: FunId) {
        let block = self.prog.funs[fun.0].entry;
        self.context_stack.push(Context { fun, block });
        self.enter_block(block);
    }

    fn leave_fun(&mut self) {
        self.context_stack.pop().expect("Function stack underflow");
    }

    fn fun(&self) -> FunId {
        self.context_stack.last().expect("Function stack underflow").fun
    }

    fn block(&self) -> BlockId {
        self.context_stack.last().expect("Function stack underflow").block
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
            None => Err(ParseError::UnexpectedError),
            Some(actual) if expected == actual => Ok(()),
            Some(actual) => Err(ParseError::UnexpectedToken(actual)),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.tokens.next() {
            Some(Token::Ident(name)) => Ok(name.clone()),
            None => Err(ParseError::UnexpectedError),
            Some(actual) => Err(ParseError::UnexpectedToken(actual)),
        }
    }

    fn push_insn(&mut self, opcode: Opcode, operands: Vec<Opnd>) -> Opnd {
        match (&opcode, &operands[..]) {
            (Opcode::Add, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Int(l+r)),
            (Opcode::Mul, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Int(l*r)),
            (Opcode::Div, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) if *r != 0 => Opnd::Const(Value::Int(l/r)),
            (Opcode::Equal, [Opnd::Const(l), Opnd::Const(r)]) => Opnd::Const(Value::Bool(l==r)),
            (Opcode::NotEqual, [Opnd::Const(l), Opnd::Const(r)]) => Opnd::Const(Value::Bool(l != r)),
            (Opcode::Greater, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Bool(l>r)),
            (Opcode::GreaterEqual, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Bool(l>=r)),
            (Opcode::Less, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Bool(l<r)),
            (Opcode::LessEqual, [Opnd::Const(Value::Int(l)), Opnd::Const(Value::Int(r))]) => Opnd::Const(Value::Bool(l<=r)),
            _ => Opnd::Insn(self.prog.push_insn(self.fun(), self.block(), opcode, operands))
        }
    }

    fn parse_program(&mut self) -> Result<(), ParseError> {
        self.enter_fun(self.prog.entry);
        let mut env = Env::new(self.prog.entry);
        while let Some(token) = self.tokens.peek() {
            if *token == Token::Eof { break; }
            self.parse_statement(&mut env)?;
        }
        self.leave_fun();
        Ok(())
    }

    fn merge(&mut self, left: Option<&Opnd>, right: Option<&Opnd>) {
        match (left, right) {
            (Some(left), Some(right)) if left == right => {}
            (Some(left), Some(right)) => { self.push_insn(Opcode::Phi, vec![left.clone(), right.clone()]); }
            (Some(value), None) | (None, Some(value)) => { self.push_insn(Opcode::Phi, vec![value.clone(), Opnd::Const(Value::Nil)]); }
            (None, None) => panic!("Cannot happen"),
        }
    }

    fn merge_envs(&mut self, left: &Env, right: &Env) {
        let mut all_keys: HashSet<&String> = HashSet::new();
        all_keys.extend(&mut left.bindings.keys());
        all_keys.extend(&mut right.bindings.keys());
        for key in all_keys {
            let left_value = left.bindings.get(key);
            let right_value = right.bindings.get(key);
            self.merge(left_value, right_value);
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
                env.define(&name, value);
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

                if self.tokens.peek() == Some(&Token::Else) {
                    self.tokens.next();
                    self.enter_block(iffalse_block);
                    let mut iffalse_env = env.clone();
                    self.parse_statement(&mut iffalse_env)?;
                    let join_block = self.new_block();
                    self.push_insn(Opcode::Branch(join_block), vec![]);

                    self.enter_block(join_block);
                    self.merge_envs(&iftrue_env, &iffalse_env);
                    self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(join_block), vec![]);
                } else {
                    self.enter_block(iffalse_block);
                    self.merge_envs(&env, &iftrue_env);
                    self.prog.push_insn(self.fun(), iftrue_end, Opcode::Branch(iffalse_block), vec![]);
                }
                return Ok(());  // no semicolon
            }
            Some(Token::LCurly) => {
                // New scope
                self.tokens.next();
                while let Some(token) = self.tokens.peek() {
                    if *token == Token::RCurly { break; }
                    self.parse_statement(&mut env)?;
                }
                return self.expect(Token::RCurly);  // no semicolon
            }
            Some(token) => { self.parse_expression(&mut env)?; Ok(()) },
            None => { Err(ParseError::UnexpectedError) }
        }?;
        self.expect(Token::Semicolon)
    }

    fn parse_function(&mut self, env: &Env) -> Result<(), ParseError> {
        let name = self.expect_ident()?;
        let fun = self.prog.push_fun(name);
        self.enter_fun(fun);
        self.expect(Token::LParen)?;
        let mut idx = 0;
        // TODO(max): We need a way to find out if a variable is from an outer context (global,
        // closure) when using it so we don't bake in the value at compile-time.
        let mut func_env = Env::from_parent(fun, &env);
        loop {
            match self.tokens.peek() {
                Some(Token::Ident(name)) => {
                    let name = name.clone();
                    func_env.define(&name, self.push_insn(Opcode::Param(idx), vec![]));
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
        Ok(())
    }

    fn parse_expression(&mut self, mut env: &mut Env) -> Result<Opnd, ParseError> {
        self.parse_(env, 0)
    }

    fn parse_(&mut self, mut env: &mut Env, prec: u32) -> Result<Opnd, ParseError> {
        let mut lhs = match self.tokens.peek() {
            None => Err(ParseError::UnexpectedError),
            Some(Token::Nil) => { let result = Opnd::Const(Value::Nil); self.tokens.next(); Ok(result) }
            Some(Token::Bool(value)) => { let result = Opnd::Const(Value::Bool(*value)); self.tokens.next(); Ok(result) }
            Some(Token::Int(value)) => { let result = Opnd::Const(Value::Int(*value)); self.tokens.next(); Ok(result) }
            Some(Token::Str(value)) => { let result = Opnd::Const(Value::Str(value.clone())); self.tokens.next(); Ok(result) }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.tokens.next();
                if self.tokens.peek() == Some(&Token::Equal) {
                    // Assignment
                    self.tokens.next();
                    // TODO(max): Operator precedence ...?
                    let value = self.parse_expression(&mut env)?;
                    match env.set(&name, value.clone()) {
                        Some(VarKind::Local) => Ok(value),
                        // TODO(max): Figure out a local numbering scheme I guess
                        Some(VarKind::Closure) => Ok(self.push_insn(Opcode::ClosureSet(0), vec![value])),
                        None => Err(ParseError::UnboundName(name)),
                    }
                } else {
                    match env.lookup(&name) {
                        Some((VarKind::Local, value)) => Ok(value),
                        // TODO(max): Figure out a local numbering scheme I guess
                        Some((VarKind::Closure, value)) => Ok(self.push_insn(Opcode::ClosureRef(0), vec![])),
                        None => Err(ParseError::UnboundName(name)),
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
            let (assoc, op_prec, opcode) = match token {
                Token::And => (Assoc::Left, 0, Opcode::Abort),  // not really Abort; just used for precedence
                Token::Or => (Assoc::Left, 0, Opcode::Abort),  // not really Abort; just used for precedence
                Token::EqualEqual => (Assoc::Left, 1, Opcode::Equal),
                Token::BangEqual => (Assoc::Left, 1, Opcode::NotEqual),
                Token::Greater => (Assoc::Left, 2, Opcode::Greater),
                Token::GreaterEqual => (Assoc::Left, 2, Opcode::GreaterEqual),
                Token::Less => (Assoc::Left, 2, Opcode::Less),
                Token::LessEqual => (Assoc::Left, 2, Opcode::LessEqual),
                Token::Plus => (Assoc::Any, 3, Opcode::Add),
                Token::Minus => (Assoc::Left, 3, Opcode::Sub),
                Token::Star => (Assoc::Any, 4, Opcode::Mul),
                Token::ForwardSlash => (Assoc::Left, 4, Opcode::Div),
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
            } else {
                let rhs = self.parse_(&mut env, next_prec)?;
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
        (1+2)*3; 4/5; 6 == 7; print 1+8 <= 9; print nil;
        fun empty() { return nil; }
        // a comment
        fun inc(a) { return a+1; }
        fun params(a, b) { }
        var x = 1;
        fun read_global() { return x; }
        if (1) {
            x = 2;
        } else {
            x = 3;
        }
        print x;
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
    fn test_string_lit() {
        check(r#""abc""#, expect![[r#"[Str("abc")]"#]])
    }

    #[test]
    fn test_unterminated_string_lit() {
        check_error("\"abc", expect!["UnterminatedStringLiteral"])
    }
}
