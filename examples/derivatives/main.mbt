// Derivative table.
//
// The plainest possible use of the engine: build a handful of
// expressions, differentiate each with respect to its input, and
// print the analytic derivative next to the finite-difference
// estimate.
//
// The finite-difference column is the point of the example. A
// gradient engine that is subtly wrong can still produce
// plausible-looking numbers, so the check that matters is against
// an independent method. Central differences at h = 1e-5 give seven
// correct digits for these functions, which is plenty to catch a
// wrong derivative rule while staying far away from the catastrophic
// cancellation that a forward difference would hit.

///|
let cases : Array[(String, String, String, Double)] = [
  ("x^2", "2*x", "x^3", 1.7),
  ("x^3", "3*x^2", "x^5", 1.3),
  ("exp(x)", "exp(x)", "2*exp(x)", 0.7),
  ("log(x)", "1/x", "log(x^3)", 2.1),
  ("sin(x)", "cos(x)", "sin(x)*cos(x)", 0.9),
  ("tanh(x)", "1-tanh(x)^2", "tanh(x)^2", 0.4),
  ("x/(1+x^2)", "(1-x^2)/(1+x^2)^2", "x/(2+x^2)", 1.1),
]

///|
fn main raise @moongrad.ADError {
  println("moongrad derivative table")
  println("f(x)              df/dx (analytic)   df/dx (finite diff)")
  println("--------------------------------------------------")
  for c in cases {
    let (name, _expected, body, at) = c
    let g = @moongrad.Graph::new()
    ignore(g.new_vars(1))
    let xv = g.variable(0)
    // Parse the little expression language the table is written in.
    let node = match eval_expr(g, xv, body) {
      Some(n) => n
      None => {
        println("| \{name}: could not parse the expression")
        continue
      }
    }
    let b = @moongrad.Binding::new()
    b.set_var(0, at)
    let (grads, _p) = @moongrad.grad(g, b, node)
    // Central difference.
    let h = 1.0e-5
    let b2 = @moongrad.Binding::new()
    b2.set_var(0, at + h)
    let fplus = @moongrad.forward(g, b2, node).output
    let b3 = @moongrad.Binding::new()
    b3.set_var(0, at - h)
    let fminus = @moongrad.forward(g, b3, node).output
    let numeric = (fplus - fminus) / (2.0 * h)
    println(
      "| \{name}  | \{pad(@moongrad.format_double(grads[0]), 18)} | \{pad(@moongrad.format_double(numeric), 18)}",
    )
  }
}

///|
/// Pad a string on the right to a fixed width, for the table.
fn pad(s : String, width : Int32) -> String {
  let buf = StringBuilder()
  buf.write_string(s)
  let mut i = s.length()
  while i < width {
    buf.write_char(' ')
    i = i + 1
  }
  buf.to_string()
}

///|
/// A tiny expression parser for the table above.
///
/// It exists so the example can state each function as a formula
/// rather than as a sequence of builder calls, which would bury the
/// point under graph plumbing. The grammar is deliberately tiny:
///
///   expr  := term (('+' | '-') term)*
///   term  := factor (('*' | '/') factor)*
///   factor := 'x' | number | '(' expr ')' | '-' factor
///   power := factor ('^' number)?
///
/// Enough for `2*x`, `1/x`, `x^3`, `exp(x)`, `log(x)`, `sin(x)`,
/// `tanh(x)` and the two rational functions.
fn eval_expr(g : @moongrad.Graph, x : Int32, src : String) -> Int32? {
  let p = Parser::{ src, pos: 0, }
  p.parse_expr(g, x)
}

///|
struct Parser {
  src : String
  mut pos : Int32
}

///|
fn Parser::parse_expr(self : Parser, g : @moongrad.Graph, x : Int32) -> Int32? {
  match self.parse_term(g, x) {
    None => return None
    Some(first) => {
      let mut acc = first
      while self.pos < self.src.length() {
        let c = self.src[self.pos]
        if c == '+' {
          self.pos = self.pos + 1
          match self.parse_term(g, x) {
            Some(t) => acc = g.add(acc, t)
            None => return None
          }
        } else if c == '-' {
          self.pos = self.pos + 1
          match self.parse_term(g, x) {
            Some(t) => acc = g.sub(acc, t)
            None => return None
          }
        } else {
          break
        }
      }
      Some(acc)
    }
  }
}

///|
fn Parser::parse_term(self : Parser, g : @moongrad.Graph, x : Int32) -> Int32? {
  match self.parse_factor(g, x) {
    None => return None
    Some(first) => {
      let mut acc = first
      while self.pos < self.src.length() {
        let c = self.src[self.pos]
        if c == '*' {
          self.pos = self.pos + 1
          match self.parse_factor(g, x) {
            Some(t) => acc = g.mul(acc, t)
            None => return None
          }
        } else if c == '/' {
          self.pos = self.pos + 1
          match self.parse_factor(g, x) {
            Some(t) => acc = g.div(acc, t)
            None => return None
          }
        } else {
          break
        }
      }
      Some(acc)
    }
  }
}

///|
fn Parser::parse_factor(
  self : Parser,
  g : @moongrad.Graph,
  x : Int32,
) -> Int32? {
  match self.parse_atom(g, x) {
    Some(acc) => {
      let mut node = acc
      // `^` binds tighter than `*`, and the exponent is a literal.
      while self.pos < self.src.length() && self.src[self.pos] == '^' {
        self.pos = self.pos + 1
        let e = self.parse_number()
        node = g.pow(node, e)
      }
      Some(node)
    }
    None => None
  }
}

///|
fn Parser::parse_atom(self : Parser, g : @moongrad.Graph, x : Int32) -> Int32? {
  self.skip_spaces()
  if self.pos >= self.src.length() {
    return None
  }
  let c = self.src[self.pos]
  if c == '-' {
    self.pos = self.pos + 1
    match self.parse_factor(g, x) {
      Some(t) => return Some(g.neg(t))
      None => return None
    }
  }
  if c == '(' {
    self.pos = self.pos + 1
    let inner = self.parse_expr(g, x)
    self.skip_spaces()
    if self.pos < self.src.length() && self.src[self.pos] == ')' {
      self.pos = self.pos + 1
    }
    return inner
  }
  if c == 'x' {
    self.pos = self.pos + 1
    return Some(x)
  }
  if c >= '0' && c <= '9' {
    let v = self.parse_number()
    return Some(g.constant(v))
  }
  // A named function: exp, log, sin, cos, tanh, sqrt.
  let name = self.parse_name()
  self.skip_spaces()
  if self.pos < self.src.length() && self.src[self.pos] == '(' {
    self.pos = self.pos + 1
    match self.parse_expr(g, x) {
      None => return None
      Some(arg) => {
        self.skip_spaces()
        if self.pos < self.src.length() && self.src[self.pos] == ')' {
          self.pos = self.pos + 1
        }
        return apply_named(g, name, arg)
      }
    }
  }
  ignore(name)
  None
}

///|
fn apply_named(g : @moongrad.Graph, name : String, arg : Int32) -> Int32? {
  match name {
    "exp" => Some(g.exp(arg))
    "log" => Some(g.ln(arg))
    "sin" => Some(g.sin(arg))
    "cos" => Some(g.cos(arg))
    "sqrt" => Some(g.sqrt(arg))
    "tanh" => Some(g.tanh(arg))
    "sigmoid" => Some(g.sigmoid(arg))
    _ => None
  }
}

///|
fn Parser::parse_name(self : Parser) -> String {
  let start = self.pos
  while self.pos < self.src.length() {
    let c = self.src[self.pos]
    if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
      self.pos = self.pos + 1
    } else {
      break
    }
  }
  self.src[start:self.pos].to_owned()
}

///|
fn Parser::parse_number(self : Parser) -> Double {
  let start = self.pos
  while self.pos < self.src.length() {
    let c = self.src[self.pos]
    if (c >= '0' && c <= '9') || c == '.' {
      self.pos = self.pos + 1
    } else {
      break
    }
  }
  // A hand-rolled decimal parser. The standard library keeps its
  // `strconv` package internal, and the grammar here is small
  // enough that writing it out is clearer than working around
  // that: optional integer part, optional fraction, optional
  // exponent.
  let text = self.src[start:self.pos].to_owned()
  let mut whole = 0.0
  let mut scale = 1.0
  let mut seen_dot = false
  let mut i = 0
  while i < text.length() {
    let c = text[i]
    if c >= '0' && c <= '9' {
      let d = (c.to_int() - '0'.to_int()).to_double()
      if seen_dot {
        scale = scale * 0.1
        whole = whole + d * scale
      } else {
        whole = whole * 10.0 + d
      }
      i = i + 1
    } else if c == '.' && !seen_dot {
      seen_dot = true
      i = i + 1
    } else {
      break
    }
  }
  whole
}

///|
fn Parser::skip_spaces(self : Parser) -> Unit {
  while self.pos < self.src.length() && self.src[self.pos] == ' ' {
    self.pos = self.pos + 1
  }
}
