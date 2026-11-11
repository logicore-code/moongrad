// aaa_summary_test.mbt
//
// The report.
//
// The file name is not decoration. MoonBit runs the test files in
// reverse alphabetical order, so a file called "aaa_..." is the last
// one to run — and the last one has to be this one, because the
// failures of every other file are collected in a shared list and
// reported from here.
//
// A summary that ran earlier would miss the failures of the files
// after it, and a test suite that cannot see its own failures is
// worse than one that has none: the checkpoint tests were failing
// for months of commits without anything going red.

///|
test "zz_summary_of_gradient_mismatches" {
  let list = failures
  for f in list {
    println("  FAIL \{f}")
  }
  assert_eq(list.length(), 0)
}
