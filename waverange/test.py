from pprint import pprint
from numpy import array


def parse_file(file_path):
    chunks = []
    current_chunk = {'q_step': None, 'min_val': None, 'data': []}

    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains two float values (indicating the start of a new chunk)
            if ' ' in line and all(s.replace('.', '', 1).replace('-', '', 1).isdigit() for s in line.split()):
                if current_chunk['q_step'] is not None:
                    # If a chunk is already being processed, append it to the chunks list
                    chunks.append(current_chunk)
                    current_chunk = {'q_step': None, 'min_val': None, 'data': []}

                q_step, min_val = map(float, line.split())
                current_chunk['q_step'] = q_step
                current_chunk['min_val'] = min_val
            else:
                # Add the line to the current chunk's data
                current_chunk['data'].append(line.strip())

        # Add the last chunk to the list (if not empty)
        if current_chunk['q_step'] is not None:
            chunks.append(current_chunk)

    return chunks

# Usage
file_path = 'compressed/mesh'  # Replace with the actual file path
parsed_data = parse_file(file_path)

# pprint(parsed_data[0])
pprint(eval(''.join(parsed_data[1]['data'][1:])))
# pprint(parsed_data[2])
# pprint(parsed_data[3])