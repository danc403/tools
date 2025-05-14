Found on a Reddit post:
TL;DR: in your llama.cpp command, add: 
-ngl 49 --override-tensor "([0-9]+).ffn_.*_exps.=CPU" --ubatch-size 1
Explanation: 
-ngl 49
• 
offload all 49 layers to GPU 
--override-tensor "([0-9]+).ffn_.*_exps.=CPU"
• 
...except for the MOE weights 
--ubatch-size 1
• 
process the prompt in batches of 1 at a time (instead of the default 512 - otherwise your SSD will be the bottleneck and prompt processing will be slower) 
This radically speeds up inference by taking advantage of LLama 4's MOE architecture. LLama 4 Maverick has 400 billion total parameters, but only 17 billion active parameters. Some are needed on every token generation, while others are only occasionally used. So if we put the parameters that are always needed onto GPU, those will be processed quickly, and there will just be a small number that need to be handled by the CPU. This works so well that the weights don't even need to all fit in your CPU's RAM - many of them can memory mapped from NVMe. 
My results with Llama 4 Maverick: 
• 
Unsloth's UD-Q4_K_XL quant is 227GB 
• 
Unsloth's Q8_0 quant is 397GB 
Both of those are much bigger than my RAM + VRAM (128GB + 3x24GB). But with these tricks, I get 15 tokens per second with the UD-Q4_K_M and 6 tokens per second with the Q8_0. 
Full llama.cpp server commands: 
Note: the --override-tensor command is tweaked because I had some extra VRAM available, so I offloaded most of the MOE layers to CPU, but loaded a few onto each GPU. 
UD-Q4_K_XL: 
./llama-server -m Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf -ngl 49 -fa -c 16384 --override-tensor "([1][1-9]|[2-9][0-9]).ffn_.*_exps.=CPU,([0-2]).ffn_.*_exps.=CUDA0,([3-6]).ffn_.*_exps.=CUDA1,([7-9]|[1][0]).ffn_.*_exps.=CUDA2" --ubatch-size 1
Q8_0: 
./llama-server -m Llama-4-Maverick-17B-128E-Instruct-Q8_0-00001-of-00009.gguf -ngl 49 -fa -c 16384 --override-tensor "([6-9]|[1-9][0-9]).ffn_.*_exps.=CPU,([0-1]).ffn_.*_exps.=CUDA0,([2-3]).ffn_.*_exps.=CUDA1,([4-5]).ffn_.*_exps.=CUDA2" --ubatch-size 1
Credit goes to the people behind Unsloth for this knowledge. I hadn't seen people talking about this here, so I thought I'd make a post. 
