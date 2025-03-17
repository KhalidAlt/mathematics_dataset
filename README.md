```markdown
# Mathematics Dataset

This dataset code generates mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.

Original paper: [Analysing Mathematical
Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)
(Saxton, Grefenstette, Hill, Kohli).

## Arabic Language Support

We have expanded this dataset to include Arabic templates for generating mathematical questions. The language can be selected using the `.env` variable **`LANG`**, which can be set to:

- **`en`** → Generate questions in English (default)
- **`ar`** → Generate questions in Arabic

To set the language, define it in your `.env` file:
```shell
LANG=ar
```
or set it dynamically when running the script:
```shell
LANG=ar python -m mathematics_dataset.generate
```

## Example Questions

### English:
```
Question: Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.
Answer: 4

Question: Calculate -841880142.544 + 411127.
Answer: -841469015.544
```

### Arabic:
```
السؤال: حل المعادلتين -42*r + 27*c = -1167 و 130*r + 4*c = 372 لإيجاد قيمة r.
الإجابة: 4

السؤال: احسب -841880142.544 + 411127.
الإجابة: -841469015.544
```

## Pre-generated Data

[Pre-generated files](https://console.cloud.google.com/storage/browser/mathematics-dataset)

### Version 1.0

This is the version released with the original paper. It contains 2 million
(question, answer) pairs per module, with questions limited to 160 characters in
length, and answers to 30 characters in length. Note the training data for each
question type is split into "train-easy", "train-medium", and "train-hard". This
allows training models via a curriculum. The data can also be mixed together
uniformly from these training datasets to obtain the results reported in the
paper. Categories:

* **algebra** (linear equations, polynomial roots, sequences)
* **arithmetic** (pairwise operations and mixed expressions, surds)
* **calculus** (differentiation)
* **comparison** (closest numbers, pairwise comparisons, sorting)
* **measurement** (conversion, working with time)
* **numbers** (base conversion, remainders, common divisors and multiples,
  primality, place value, rounding numbers)
* **polynomials** (addition, simplification, composition, evaluating, expansion)
* **probability** (sampling without replacement)

## Getting the Source

### PyPI

The easiest way to get the source is to use pip:

```shell
$ pip install mathematics_dataset
```

### From GitHub

Alternatively, you can get the source by cloning the **mathematics_dataset** repository:

```shell
$ git clone https://github.com/deepmind/mathematics_dataset
$ pip install --upgrade mathematics_dataset/
```

## Running the Code

To generate examples, use the `generate` script:

```shell
python -m mathematics_dataset.generate --filter=linear_1d
```

To generate Arabic questions, ensure **`LANG=ar`** is set:

```shell
LANG=ar python -m mathematics_dataset.generate --filter=linear_1d
```

For writing the generated examples to text files, use `generate_to_file.py`:

```shell
python generate_to_file.py
```

This can be adapted for training needs.

## Acknowledgment

This work builds upon DeepMind's **Mathematics Dataset**, expanding its capabilities with multilingual support. We acknowledge and appreciate DeepMind's foundational contributions to this field.

## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Mathematics Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/mathematics_dataset</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/mathematics_dataset</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">This dataset consists of mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://identifiers.org/arxiv:1904.01557</code></td>
  </tr>
</table>
</div>
```

---
