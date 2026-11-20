// The WebAssembly system interface, as far as moongrad's wasm build
// needs it.
//
// The wasm module imports exactly one function — fd_write, which is
// how a program prints — and exports its memory and its entry point.
// So the shim is small enough to read, and small enough to trust:
//
//   fd_write(fd, iovs, iovs_len, nwritten)
//
//     iovs points at an array of (pointer, length) pairs describing
//     the bytes to write; nwritten is where the number of bytes
//     written goes.
//
// The same file runs under Node and in a browser. The only difference
// between them is how the bytes of the module are obtained, and that
// is the caller's business.

/**
 * Make the import object the module wants, and collect what it
 * prints.
 *
 * The returned object has a `text` field, which is everything the
 * program wrote, and a `memory` setter, because the module's memory
 * is only reachable once it has been instantiated.
 */
export function makeWasi() {
  let memory = null
  const decoder = new TextDecoder()
  let text = ''

  const fdWrite = (fd, iovs, iovsLen, nwritten) => {
    if (memory === null) {
      return 8 // WASI's EBADF
    }
    const view = new DataView(memory.buffer)
    let written = 0
    for (let i = 0; i < iovsLen; i++) {
      const at = iovs + i * 8
      const pointer = view.getUint32(at, true)
      const length = view.getUint32(at + 4, true)
      text += decoder.decode(new Uint8Array(memory.buffer, pointer, length))
      written += length
    }
    if (nwritten !== 0) {
      view.setUint32(nwritten, written, true)
    }
    return 0
  }

  return {
    imports: {
      wasi_snapshot_preview1: {
        fd_write: fdWrite,
      },
    },
    setMemory: (m) => {
      memory = m
    },
    getText: () => text,
  }
}

/**
 * Run a compiled module and return what it printed.
 *
 * `bytes` is the WebAssembly module. The module is instantiated, the
 * memory is handed to the shim, and the entry point is called.
 */
export async function runWasm(bytes) {
  const wasi = makeWasi()
  const module = await WebAssembly.compile(bytes)
  const instance = await WebAssembly.instantiate(module, wasi.imports)
  wasi.setMemory(instance.exports.memory)
  instance.exports._start()
  return wasi.getText()
}
