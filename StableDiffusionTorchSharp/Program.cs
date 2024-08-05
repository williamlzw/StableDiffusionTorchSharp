using Microsoft.ML.Tokenizers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StableDiffusionTorchSharp
{
	public class Program
	{
		public static void Main()
		{
			Generate();
		}

		public static Tensor GetTimeEnmedding(float timestep)
		{
			var freqs = torch.pow(10000, -torch.arange(0, 160, dtype: torch.float32) / 160);
			var x = torch.tensor(new float[] { timestep }, dtype: torch.float32)[torch.TensorIndex.Colon, torch.TensorIndex.None] * freqs[torch.TensorIndex.None];
			return torch.cat(new Tensor[] { torch.cos(x), torch.sin(x) }, dim: -1);
		}

		public static List<int> Tokenize(Tokenizer tokenizer, string text, int maxTokens = 77)
		{
			int startToken = 49406;
			int endToken = 49407;
			var res = tokenizer.Encode(text);
			var tokens = new[] { startToken }.Concat(res.Ids.Concat(Enumerable.Repeat(0, maxTokens - res.Ids.Count - 2))).Concat(new[] { endToken }).ToArray();
			return new List<int>(tokens);
		}

		public static void Generate()
		{
			using (torch.no_grad())
			{
				int num_inference_steps = 20;
				var device = torch.device("cuda");
				float cfg = 5.5f;
				ulong seed = (ulong)new Random().Next(0, int.MaxValue);
				string modelname = @".\model\sunshinemix_PrunedFp16.safetensors";
				torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);

				Console.WriteLine("Device:" + device);
				Console.WriteLine("CFG:" + cfg);
				Console.WriteLine("Seed:" + seed);
				Console.WriteLine("Loading clip......");
				var clip = new CLIP();
				clip.load(modelname);
				clip = clip.to(device);
				clip.eval();

				Console.WriteLine("Loading unet......");
				var diffusion = new Diffusion();
				diffusion.load(modelname);
				diffusion = diffusion.to(device);
				diffusion.eval();

				Console.WriteLine("Loading vae......");
				var decoder = new Decoder();
				decoder.load(modelname);
				decoder = decoder.to(device);
				decoder.eval();

				ScalarType clipType = clip.embedding.token_embedding.weight.dtype;
				ScalarType diffusionType = diffusion.final.groupnorm.weight.dtype;
				ScalarType decoderType = ((Conv2d)decoder.children().First()).weight.dtype;

				string VocabPath = @".\model\vocab.json";
				string MergesPath = @".\model\merges.txt";
				var tokenizera = new TokenizerCustom(VocabPath, MergesPath);//custom Tokenizer
				var tokenizer = new Tokenizer(new Bpe(VocabPath, MergesPath, endOfWordSuffix: "</w>"));

				string prompt = "typographic art bird. stylized, intricate, detailed, artistic, text-based";
				string uncond_prompts = "";

				Console.WriteLine("Clip is doing......");
				var cond_tokens_ids = Tokenize(tokenizer, prompt);
				var cond_tokens_idsa = tokenizera.Encode(prompt);//custom Tokenizer
				var cond_tokens = torch.tensor(cond_tokens_ids, torch.@long).unsqueeze(0).to(device);
				var cond_context = clip.forward(cond_tokens);

				var uncond_tokens_ids = Tokenize(tokenizer, uncond_prompts);
				var uncond_tokens = torch.tensor(uncond_tokens_ids, torch.@long).unsqueeze(0).to(device);
				var uncond_context = clip.forward(uncond_tokens);

				var context = torch.cat(new Tensor[] { cond_context, uncond_context }).to(diffusionType).to(device);

				Console.WriteLine("Getting latents......");
				long[] noise_shape = new long[] { 1, 4, 64, 64 };
				Generator generator = new Generator(seed, device);
				var latents = torch.randn(noise_shape, generator: generator);
				latents = latents.to(diffusionType).to(device);
				var sampler = new EulerDiscreteScheduler();

				sampler.SetTimesteps(num_inference_steps, device);
				latents *= sampler.InitNoiseSigma();

				for (int i = 0; i < num_inference_steps; i++)
				{
					Console.WriteLine($"begin step");
					var timestep = sampler.timesteps_[i];
					var time_embedding = GetTimeEnmedding(timestep.ToSingle()).to(diffusionType).to(device);
					var input_latents = sampler.ScaleModelInput(latents, timestep);
					input_latents = input_latents.repeat(2, 1, 1, 1).to(diffusionType).to(device);
					var output = diffusion.forward(input_latents, context, time_embedding);
					var ret = output.chunk(2);
					var output_cond = ret[0];
					var output_uncond = ret[1];
					output = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(output, timestep, latents);
					Console.WriteLine($"end step, {i}");
				}

				latents = latents.to(decoderType);
				Console.WriteLine($"begin decoder");
				var images = decoder.forward(latents);
				Console.WriteLine($"end decoder");
				images = images.clip(-1, 1) * 0.5 + 0.5;
				images = images.cpu();
				images = torchvision.transforms.functional.convert_image_dtype(images, torch.ScalarType.Byte);
				torchvision.io.write_jpeg(images, "result.jpg");

			}
		}
	}
}