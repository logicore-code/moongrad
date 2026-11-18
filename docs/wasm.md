# Running moongrad on the web

An automatic differentiation engine is worth writing in MoonBit mostly
because of where it can run. A gradient compiler that needs a Python
process, a JIT, and a runtime with a garbage collector attached cannot
go into a browser tab; this one can, because it is a MoonBit library
with no FFI and nothing outside the standard library.

The same source builds for four targets:

| Target | What it is for | The `reuse` example comes out as |
|---|---|---|
| `native` | the fastest, for training runs and servers | a native executable |
| `wasm` | a browser or a serverless function | 264 KB of WebAssembly |
| `wasm-gc` | the same, with the GC runtime in the module | 152 KB of WebAssembly |
| `js` | a Node process, or a bundler that wants JavaScript | 336 KB of JavaScript |

## Building it

```bash
moon build --target wasm examples/reuse
```

The artifact is `_build/wasm/release/examples/reuse/reuse.wasm`. There
is no JavaScript glue to ship with it: the module carries its own
runtime.

## Running it

`moon run` will execute a wasm build under Node, which is the quickest
way to check that a change behaves the same on both targets:

```bash
moon run examples/reuse --target wasm
```

The output is identical to the native run, down to the last digit of
the disagreement between the analytic gradient and the finite
difference — `8.386678018723615e-10` on both. That matters more than
it sounds: a floating-point kernel that reorders its operations for a
different target would show up here as a difference in the last bits,
and the gradient checks in the test suite would still pass on the
target they were written for.

The CI workflow runs two examples on the wasm target for exactly this
reason.

## Putting it in a page

The wasm module exports `_start`, so a page needs three things: the
`.wasm` file served with the `application/wasm` content type, a
`WebAssembly.instantiate` call, and somewhere for the output to go.

MoonBit's wasm target writes to stdout rather than to a JavaScript
value, so the usual shape is a WASI shim on the JavaScript side — the
`@wasmer/wasi` package, or any of the small WASI polyfills, will do:

```js
import { WASI } from '@wasmer/wasi'
import { readFile } from 'node:fs/promises'

const wasi = new WASI()
const bytes = await readFile('reuse.wasm')
const module = await WebAssembly.compile(bytes)
const instance = await WebAssembly.instantiate(module, wasi.getImportObject())
wasi.start(instance)
```

The same file, served from the same place, runs in a browser tab with
no server behind it. A model whose parameters are fitted on a server
and shipped as JSON — `model_io.mbt` turns a model into a JSON string
and back, and the round trip is exact down to the sign of a zero —
can be differentiated in the tab, with the gradient program compiled
there rather than shipped alongside. Writing the string to a file is
the caller's job; the library reads no files at all.

## What this does not buy you

The wasm build is not fast, and it is not small. A tensor operation in
this library is a loop over a flat array in MoonBit, which is fine on
a native target and slow in a browser; the native build is roughly an
order of magnitude faster on the convolution in `examples/cnn`. What
the wasm build buys is that the engine *is there* — no FFI, no Python,
no second language, one source file compiled by one command.

The library is not trying to be a numerical kernel. The value it
offers is that the function being differentiated is user-supplied
rather than MoonBit code the compiler could have optimised directly,
and that the whole thing — the graph, the compiler, the optimiser,
the Hessian — is one dependency with no runtime attached.
