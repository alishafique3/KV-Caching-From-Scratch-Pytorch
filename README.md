# KV Caching From Scratch Pytorch

Last week, I explored KV caching from scratch while working on a GPT-2 model. I ran my experiments on a Colab T4 GPU to better understand how caching improves inference speed in large language models.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺?
In autoregressive generation, LLMs generate one token at a time, and each new token has to attend to all previous tokens.
So if your model generated: "White → Fluffy → Cat"
…the attention block still recomputes the Keys and Values for "White" and "Fluffy" every time.

That’s a lot of unnecessary computation, especially as the output grows longer.

𝗧𝗵𝗲 𝗞𝗩 𝗖𝗮𝗰𝗵𝗲 𝗦𝗼𝗹𝘂𝘁𝗶𝗼𝗻:
I implemented a caching mechanism where:
 • The model caches the Keys & Values for input tokens during prefill.
 • For each new token, it only computes the K/V for that token.
 • Previous tokens just pull from the cache, no recompute needed.

📊 𝗥𝗲𝘀𝘂𝗹𝘁𝘀:
Tested on a Colab T4 GPU using different output lengths:
441 → ~2.7×
761 → ~4.6×
1001 → ~6.7×

![KV Cache](https://github.com/user-attachments/assets/33c5ce39-fd43-4c74-9f01-daac0afa13a0)


𝗢𝗻𝗲 𝗙𝘂𝗻 𝗜𝗻𝘀𝗶𝗴𝗵𝘁:
For shorter outputs, KV caching doesn't always help.
In my tests, device communication overhead on CUDA sometimes outweighed the gains for small models like GPT-2.

Shoutout to Sebastian Raschka, PhD for his amazing blog post on KV Caching
