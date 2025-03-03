import csv
import os
import time

import openai

# SwissAI API Configuration
client = openai.Client(api_key="sk-rc-75HStaTc3UOSoVgyXSEU7w", base_url="https://fmapi.swissai.cscs.ch")

BATCH_SIZE = 10  # Number of lines per batch


def evaluate_batch_quality(texts: list[str], model_name: str = "meta-llama/Llama-3.3-70B-Instruct") -> list[bool]:
    """
    Uses SwissAI API to evaluate a batch of texts.

    Args:
        texts (list[str]): List of text samples to evaluate.
        model_name (str): SwissAI model to use for evaluation.

    Returns:
        list[bool]: List of True/False values indicating if each text is useful.
    """
    try:
        eval_prompt = "Determine if the following texts are useful for training an LLM on Wikipedia fact updates.\n\n"

        for i, text in enumerate(texts):
            eval_prompt += f"{i+1}. {text}\n"

        eval_prompt += "\nIf the text is informative and fact-based, return 'true'. If the text is meaningless, misleading, or repetitive, return 'false'. Return only 'true' or 'false' do not number the responses and give as many as there are texts.\n"

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"content": eval_prompt, "role": "user"}],
            max_tokens=5 * len(texts),  # Adjust max tokens to accommodate batch
            stream=False,
        )

        results = response.choices[0].message.content.strip().lower().split("\n")
        return [res.strip() == "true" for res in results]

    except Exception as e:
        print(f" Error evaluating batch data quality with SwissAI: {e}")
        return [False] * len(texts)  # Default to False if API request fails


def filter_csv(input_csv: str, output_csv: str):
    """
    Reads a CSV file, filters lines in batches based on SwissAI evaluation, and writes a new cleaned CSV.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the cleaned CSV file.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file '{input_csv}' not found.")
        return

    start_time = time.time()  # Start timer

    with (
        open(input_csv, newline="", encoding="utf-8") as infile,
        open(output_csv, "w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        total_lines = 0
        kept_lines = 0
        removed_lines = 0

        batch_texts = []
        batch_rows = []

        for row in reader:
            if not row:  # Skip empty rows
                continue

            text = " ".join(row)
            batch_texts.append(text)
            batch_rows.append(row)

            if len(batch_texts) == BATCH_SIZE:  # Process batch
                results = evaluate_batch_quality(batch_texts)
                for idx, keep in enumerate(results):
                    if idx >= len(batch_rows):
                        break
                    if keep:
                        # writer.writerow(batch_rows[idx])
                        kept_lines += 1
                    else:
                        removed_lines += 1

                total_lines += len(batch_texts)
                batch_texts, batch_rows = [], []

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n✅ Filtering completed in {total_time:.2f} seconds.")
        print(f" Kept {kept_lines}/{total_lines} lines, Removed {removed_lines}.")
        print(f" Average time per line: {total_time / total_lines:.4f} seconds.")

        # ADDED: Returning statistics for benchmarking
        return total_lines, kept_lines, removed_lines, total_time


def run_benchmark(
    input_csv_path, output_csv_path, iterations=5, dynamic_batch_size=None, results_file="benchmark_results.txt"
):
    """
    Runs the filter_csv function multiple times, collects stats, and writes them to a file.

    Args:
        input_csv_path (str): The path to the input CSV.
        output_csv_path (str): The path for the filtered output CSV.
        iterations (int): Number of times to run filter_csv.
        dynamic_batch_size (int or None): If provided, overrides the global BATCH_SIZE for each run.
        results_file (str): File to write benchmark results to.
    """
    # If a new batch size is specified, override the global BATCH_SIZE
    if dynamic_batch_size is not None:
        global BATCH_SIZE
        BATCH_SIZE = dynamic_batch_size
        # print(f"\n🔹 Using dynamic batch size: {BATCH_SIZE}")

    results = []
    for i in range(1, iterations + 1):
        # print(f"\n--- Benchmark Run {i} ---")
        total, kept, removed, duration = filter_csv(input_csv_path, output_csv_path)

        results.append(
            {
                "Iteration": i,
                "Total_Lines": total,
                "Kept_Lines": kept,
                "Removed_Lines": removed,
                "Duration_s": round(duration, 3),
            }
        )

    # Instead of printing to console, build a table string and write to file
    table_str = []
    table_str.append("Benchmark Results:")
    table_str.append(" Iter |  Total  |  Kept  | Removed |  Duration (s)")
    table_str.append("------|---------|--------|---------|--------------")

    total_duration = 0.0
    for r in results:
        row_str = (
            f"  {r['Iteration']:>2}  | {r['Total_Lines']:>7} "
            f"| {r['Kept_Lines']:>6} | {r['Removed_Lines']:>7} "
            f"| {r['Duration_s']:>12}"
        )
        table_str.append(row_str)
        total_duration += r["Duration_s"]

    avg_duration = total_duration / iterations
    table_str.append(f"\nAverage Duration over {iterations} runs: {avg_duration:.2f} seconds.\n")

    # Write results table to file
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("\n".join(table_str))


if __name__ == "__main__":
    # Original example paths
    input_csv_path = "/scratch/sjohn/temporalwiki/TWiki_Diffsets/readablediffset/0808.csv"
    output_csv_path = "cleaned_training_data.csv"
    batches = [1, 5, 10, 20, 50]
    print("\n🔹 Running LLM-Based Data Filtering with Benchmarking...")
    for batch in batches:
        run_benchmark(
            input_csv_path,
            output_csv_path,
            iterations=5,
            dynamic_batch_size=batch,  # Example: override default BATCH_SIZE=10 with 20
            results_file="my_benchmark.txt",  # Example: results written to this file
        )
        print("\n✅ Data filtering benchmark completed!")
