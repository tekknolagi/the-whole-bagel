# A little compiler project

## Goals

* AOT compiler that bundles IR so code can also be specialized at run-time (JIT)
* Little target language with enough features to be interesting
* Good test infrastructure

## Non-goals

* Massive existing language

## Thoughts

* Potential source languages
  * ChocoPy
  * COOL
  * MiniJava
  * Decaf
  * GoLite
  * Xi
  * Lox
* Desired language features
  * Structs
  * Static typing
  * Monomorphization of functions by type
  * Garbage collection
  * Userland annotations for specialization based on parameter values
* Use Chris Fallin's register allocator verifier
* Run the program until some userland `freeze` call and *then* AOT?
  * Kind of like JIT, but only once, more time to optimize
  * Kind of like wizer
* Add weval so that people can write interpreters and get method JITs "for free"

## Plan

* Create an AOT compiler that gives no thought to performance of generated code
  * Parser
  * IR
  * Register allocation
  * Code generation
  * Make all calls indirect, kind of like the PLT. This will allow us to swap
    out the code at run-time.
  * Optional interprocedural SCCP like in Loupe
* Add runtime support for profiling execution
  * Number of times a function is called
  * Branch prediction
  * Find monomorphic attribute lookups
  * Find monomorphic call sites
  * Find monomorphic parameter values
* Use existing AOT compiler for JIT
  * Enable inlining
  * Think about how to do deopt into AOT code
* Rewrite register allocator
  * AOT: Graph coloring?
  * JIT: Linear scan?
  * For both: Ian Rogers' https://arxiv.org/pdf/2011.05608
