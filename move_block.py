import sys

with open('docs/writing/thesis/ch4_evaluation.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start and end of the block
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith('\section{System Under Test}'):
        start_idx = i
        break

if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if line.startswith('\section{Detection Evaluation}'):
            # The block ends right before Detection Evaluation
            end_idx = i
            break
        # Better: let's just find \section{Detection Evaluation}
        
for i, line in enumerate(lines):
    if line.startswith('\section{Detection Evaluation}'):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    # extract the block, leaving out trailing empty lines before end_idx if we want
    block = lines[start_idx:end_idx]
    
    # replace \section with \subsection
    block[0] = block[0].replace('\section{System Under Test}', '\subsection{System Under Test}')
    
    # remove the block from the original place
    del lines[start_idx:end_idx]
    
    # find where to insert it: right after \label{sec:eval-sequences}
    insert_idx = -1
    for i, line in enumerate(lines):
        if '\label{sec:eval-sequences}' in line:
            insert_idx = i + 1
            break
            
    if insert_idx != -1:
        # insert an empty line before the block if needed
        lines[insert_idx:insert_idx] = ['\n'] + block
        
        with open('docs/writing/thesis/ch4_evaluation.tex', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Moved block successfully.')
    else:
        print('Could not find insert index.')
else:
    print('Could not find block.')
