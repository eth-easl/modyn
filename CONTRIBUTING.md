# Contributing

Please read this guide before submitting a pull request. It should give you a basic overview of some coding conventions,
style guide, and PR workflow. This document is partially inspired by (and copied
from) [Darwin](https://github.com/hpides/darwin), which is inspired by [Hyrise](https://github.com/hyrise/hyrise/blob/master/CONTRIBUTING.md).

## General

- Choose concise but descriptive class/method/variable names. Comments should be used to explain the concept and usage
  of classes and the structure of the algorithms (in the implementation).
- Use `assert` wherever it makes sense.
- Just as your code is not perfect, neither is the code that people wrote before you. Try to improve it as part of your
  PRs and do not hesitate to ask if anything is unclear. Chances are that it can be improved.
- Test your code. This includes unit *and* system tests. Try to isolate parts that can be independently tested.

## Formatting and Naming

* Much of this is/should be enforced by flake8 and pytest in CI. We are currently working on this.
* Use `autopep8` and check `flake8` before committing your code.
* Choose clear and concise names, and avoid, e.g., `i`, `j`, `ch_ptr`, unless you are using it as a loop variable in a
  short loop.
* Formatting details: 120 columns, comments above code.
* Use empty lines to structure your code.
* Naming conventions:
    * Files: lowercase separated by underscores, e.g., `score_selector.py`, usually corresponding to a class, e.g.,
      `ScoreSelector`.
    * Types (classes, structs, enums, typedefs, using): PascalCase starting with uppercase letter, e.g., `AbstractNode`.
    * Variables: lowercase separated by underscores, e.g., `data_size`.
    * Functions: lowercase separated by underscores, e.g., `get_data_size()`.
    * Private / protected members: like variables / functions with leading underscore, e.g., `_data_size`.
    * If an identifier contains a verb or an adjective in addition to a noun, the schema [verb|adjective]\[noun] is
      preferred, e.g., use `left_input` rather than ~~`input_left`~~ and `set_left_input()` rather than
      ~~`set_input_left()`~~.
* Maintain correct orthography and grammar. Comments should start with a capital letter, sentences should be finished
  with a full stop.
* Set your editor to add a newline at the end of each file.

## Branches

* `<github-name>/feature/<issue-id><name>` for features.
* `<github-name>/bug/<issue-id><name>` for bugs.
* `<github-name>/<something-something>` for small, untracked things (fixing typos, etc.).
* Example: `MaxiBoether/feature/#37-pylint-compliance`.

## Pull Requests

### Opening PRs

* If your PR is related to an existing issue, reference it in the PR's description (e.g., `fixes #123` or `refs #123`).
* Please add a description of your work in the PR.
* If your PR is not ready for a review yet, make it a draft PR.
* If you create a PR and are ready for the review, select reviewers.
* If your PR generates new code (e.g. adds new Operator), please give an example of the generated code in the
  description.
* If your PR has been reviewed, needed some changes, and should be re-reviewed, make sure to re-request a review.
* If you author a PR, you merge it. When all tests pass and you have the needed approvals, you can merge the PR.
* When merging your PR, copy your PR description (excluding the benchmark results) into the commit message. The commit
  message of the squash merge should NOT contain the individual commit messages from your branch.

### Reviewing PRs

* Keep the whole picture in mind. Often, it makes sense to make two passes: One for the code style and line-level
  modifications; one for checking how it fits into the overall picture.
* Check if the PR sufficiently adds tests both for happy and unhappy cases.
* Do not shy away from requesting changes on surrounding code that was not modified in the PR. Remember that after a PR,
  the code base should be better than before.
* Verify the CI results, including how the reported coverage changed, and (when applicable) check if performance
  have been negatively affected.