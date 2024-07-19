using TorchSharp;
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

		public static void Generate()
		{
			using (torch.no_grad())
			{
				string modelname = @".\model\sunshinemix_PrunedFp16.safetensors";
				int num_inference_steps = 15;
				var device = torch.device("cuda");
				float cfg = 7.5f;
				//ulong seed = (ulong)(randn_float() * ulong.MaxValue);
				ulong seed = 1337;

				torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);

				Console.WriteLine("Loading clip model......");
				var clip = new CLIP().half();

				clip.load(modelname);
				clip = clip.to(device);
				clip.eval();
				Console.WriteLine("Clip model loaded.");

				Console.WriteLine("Clip the prompt.");

				string VocabPath = @".\model\vocab.json";
				string MergesPath = @".\model\merges.txt";
				var tokenizer = new Tokenizer(VocabPath, MergesPath);

				string prompt = "typographic art bird. stylized, intricate, detailed, artistic, text-based.";
				string uncond_prompts = "";

				var cond_tokens_ids = tokenizer.Encode(prompt);
				var cond_tokens = torch.tensor(cond_tokens_ids, torch.@long).unsqueeze(0).to(device);
				var cond_context = clip.forward(cond_tokens);

				var uncond_tokens_ids = tokenizer.Encode(uncond_prompts);
				var uncond_tokens = torch.tensor(uncond_tokens_ids, torch.@long).unsqueeze(0).to(device);
				var uncond_context = clip.forward(uncond_tokens);

				var context = torch.cat(new Tensor[] { cond_context, uncond_context });

				//torch.save(context, "context.dat");
				//var context = torch.load("context.dat");
				Console.WriteLine("Clip done.");

				Console.WriteLine("Loading diffuson model......");
				var diffusion = new Diffusion().half();
				diffusion.load(modelname);
				diffusion = diffusion.to(device);
				diffusion.eval();
				Console.WriteLine("Loading diffuson model......");
				long[] noise_shape = new long[] { 1, 4, 64, 64 };
				Generator generator = new Generator(seed: seed, device: device);
				var latents = torch.randn(noise_shape, generator: generator).to(device);
				var sampler = new EulerDiscreteScheduler();
				Console.WriteLine("Diffuson model loaded");

				Console.WriteLine("Generate begin......");
				sampler.SetTimesteps(num_inference_steps, device);
				latents *= sampler.InitNoiseSigma();

				for (int i = 0; i < num_inference_steps; i++)
				{
					Console.WriteLine($"begin step");
					var timestep = sampler.timesteps_[i];
					var time_embedding = GetTimeEnmedding(timestep.ToSingle()).half().to(device);
					var input_latents = sampler.ScaleModelInput(latents, timestep);
					input_latents = input_latents.repeat(2, 1, 1, 1).half().to(device);
					var output = diffusion.forward(input_latents, context, time_embedding);
					var ret = output.chunk(2);
					var output_cond = ret[0];
					var output_uncond = ret[1];
					output = cfg * (output_cond - output_uncond) + output_uncond;
					latents = sampler.Step(output, timestep, latents);
					//torch.save(latents, $"latent{i}.dat");
					Console.WriteLine($"end step, {i}");
				}

				//var latents = torch.load("latent14.dat").half().to(device);

				var decoder = new Decoder().half();
				decoder.load(modelname);
				decoder = decoder.to(device);
				decoder.eval();
				Console.WriteLine($"begin decoder");
				var images = decoder.forward(latents.half().to(device));
				Console.WriteLine($"end decoder");
				images = images.clip(-1, 1) * 0.5 + 0.5;
				images = images.cpu();
				images = torchvision.transforms.functional.convert_image_dtype(images, torch.ScalarType.Byte);
				torchvision.io.write_jpeg(images, "result.jpg");
				Console.WriteLine("Image save to result.jpg");
			}
		}
	}
}