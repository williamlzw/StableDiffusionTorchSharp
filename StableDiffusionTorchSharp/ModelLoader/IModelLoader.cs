﻿namespace StableDiffusionTorchSharp.ModelLoader
{
	public interface IModelLoader
	{
		List<Tensor> ReadTensorsInfoFromFile(string fileName);
		byte[] ReadByteFromFile(Tensor tensor);

	}
}
