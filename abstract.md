# A sketch of a new VM

Pick a random Java-like programming language. [Lox][lox], for example. It has
classes, inheritance, dynamic dispatch, and garbage collection. Most
implementations are bytecode compilers and interpreters because that's the path
the book recommends. Let's make an ahead-of-time compiler instead.

Write an ahead-of-time compiler in a language like Rust. Separate the frontend
from the rest of the compiler (optimizer and backend). Bundle that backend with
the compiled user code. At run-time, re-compile hot functions using profile
information. This requires keeping the compiler IR around for every function.

It gives guest language programmers flexibility: they can pick between:

* fast AOT compile (little to no optimization)
* excellent AOT performance (more optimization, maybe interprocedural static analysis)

and between:

* more security, more stable peak performance, smaller binary (no JIT)
* higher peak perf, maybe a little harder to predict (JIT)

That's neat. It's kind of like RPython, the language used to implement PyPy, in
this regard. Maybe also a little like Julia too.

Now consider writing a bytecode interpreter in Lox (Bagel?). Bagel could be
reasonably fast, given a decent compiler, but not as fast as one written in a
language like C and compiled with an advanced compiler such as Clang.

That's unfortunate, given that we have a compiler already bundled with the
binary. We should be able to use it! We could try to expose the IR builder as a
library, but compiler internals may change over time and moving targets don't
make for attractive APIs.

## Weval

Consider another alternative: Weval/Rufus, [hopefully] accepted at PLDI 2025,
presents a relatively generic algorithm for partially evaluating bytecode
interpreters (in some SSA IR) with their bytecode programs (some read-only
memory) into compiled artifacts in the same SSA IR:

```
weval : interpreter_cfg -> bytecode -> compiled_bytecode_cfg
```

The bare minimum weval transform requires only annotating some user-defined
*context* inside the interpreter function. Normally, this context is the *pc*
variable that tracks the current instruction in the bytecode program.

The transform unravels the interpreter CFG by "running" it on the bytecode
program, merging abstract states at identical *pc* values. This context merging
is also why the partial evaluation terminates.

Adding weval to the Lox compiler means that any programmer who writes an
annotated interpreter in Lox gets a JIT "for free"; we can apply weval to Lox
IR.

Astute readers will realize that this too is similar to RPython; annotated
interpreters written in RPython also get JITs "for free". However, the RPython
JIT only works on traces; this proposed project would compile entire methods at
a time.

## More annotations

Let us continue to follow in RPython's footsteps. RPython provides all manner
of optional annotations for providing more guest language semantic information
to their advanced compiler. Is field `foo` on object `obj` read-only after it
is set? Great, mark it so, and its value can be cached across method calls even
if `obj` escapes.

As another example, it is possible to annotate RPython (Lox) functions that are
not interpreters. Such annotations include "monomorphize this function on the
type of the second parameter" and "this function is pure and results can be
cached".
