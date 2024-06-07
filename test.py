import os
def psnr_checker(psnr_path):
    with open(psnr_path, 'r') as file:
        content = file.read()

    content = content.replace(': 0', ': ')
    content = content.replace(': 0', ': 0o')
    # Parse the content into a dictionary
    psnr_dict = eval(content)

    # Create a list of PSNR values corresponding to the indices
    psnr_list = [psnr_dict.get(str(i), None) for i in range(1, len(psnr_dict) + 1)]
    print(sum(psnr_list)/129)

# Read the file
def sel_checker(so_path):
    with open(so_path, 'r') as file:
        content = file.read()

# Count the occurrences of '1' in the file
    count_of_ones = content.count('1')

    print(f"Number of '1's in the file: {count_of_ones}")

#psnr_checker("./results/test/psnr_records.txt")
save_dir = "./results/test/sel_map"
for a in range(26):
    sel_checker(os.path.join(save_dir, f"s_{int(a+1):03d}.txt"))
