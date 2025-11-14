import argparse
import sys

import yaml
from pprint import pprint
from prompt_builders import *
from tasks import *
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')


def get_full_output_path(output_path: str) -> str:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(curr_dir, output_path)
    if full_output_path[-1] == '/':
        full_output_path = full_output_path[:-1]
    return full_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='text_simplification/config_text_simplification.yaml')
    parser.add_argument('--output_path', '-o', default='text_simplification/generated_prompts/')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    args.output_path = get_full_output_path(args.output_path)
    config['args'] = {'config': args.config, 'output_path': args.output_path}

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_path, 'logs')).mkdir(parents=True, exist_ok=True)

    pprint(config)
    task = load_task(config)

    prompt_builder_zero_shot = ZeroShotPromptBuilder(config, task)
    prompt_builder_few_shot = FewShotPromptBuilder(config, task)
    prompt_builder_instruction_induction = InstructionInductionPromptBuilder(config, task)
    prompt_builder_mixed = MixedInstructionInductionPromptBuilder(config, task)
    prompt_builder_optimized = InstructionOptimizationPromptBuilder(config, task)

    time_now_str = datetime.datetime.now().strftime('%m_%d_%H-%M-%S')
    num_shot = config['main']['few_shot_num_examples']

    prompt_builder_instruction_induction = InstructionInductionPromptBuilder(config, task)

    print(prompt_builder_instruction_induction.induce_instructions(
        input_texts=["The cat sat on the mat.", "The quick brown fox jumps over the lazy dog."],
        output_texts=["A cat is sitting on a mat.", "A fast brown fox leaps over a lazy dog."]
    ))
