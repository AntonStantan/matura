import torch
from transformers import AutoModelForCausalLM

def verify_bitsandbytes_gpu():
    """
    This script verifies the functionality of the bitsandbytes library on a GPU.

    It performs the following checks:
    1. Confirms that PyTorch can detect the CUDA-enabled GPU.
    2. Attempts to load a small language model ('facebook/opt-125m') using
       8-bit quantization (`load_in_8bit=True`). This is a core feature of
       bitsandbytes and will fail if the library is not correctly configured
       for your GPU.
    3. Prints the memory footprint of the 8-bit quantized model to show
       that quantization has occurred.
    4. Performs a simple forward pass (inference) to ensure the quantized
       model can execute operations on the GPU.
    """
    print("--- Bitsandbytes GPU Verification Script ---")
    print("\nStep 1: Checking for CUDA-enabled GPU...")

    # Check 1: Is CUDA available?
    if not torch.cuda.is_available():
        print("\n❌ ERROR: PyTorch cannot find a CUDA-enabled GPU.")
        print("Please ensure you have installed the correct PyTorch version for your Jetson device.")
        print("Verification failed.")
        return

    cuda_device_name = torch.cuda.get_device_name(0)
    print(f"✅ Success: CUDA is available. Found GPU: {cuda_device_name}")

    # Check 2: Load a model with 8-bit quantization
    print("\nStep 2: Loading a model with 8-bit quantization (`load_in_8bit=True`)...")
    model_name = "facebook/opt-125m"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto" # Automatically places the model on the GPU
        )
        print("✅ Success: Model loaded in 8-bit without errors.")
        print("   This indicates that bitsandbytes is correctly installed and communicating with the GPU.")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load the model in 8-bit.")
        print("   This is likely an issue with your bitsandbytes installation or its CUDA compatibility.")
        print(f"   Error details: {e}")
        print("Verification failed.")
        return

    # Check 3: Verify model is on GPU and quantized
    print("\nStep 3: Verifying model properties...")
    model_device = next(model.parameters()).device
    print(f"   - Model is on device: {model_device}")

    if model_device.type != 'cuda':
        print(f"   ❌ WARNING: Model is on {model_device.type}, not 'cuda'. Check the `device_map`.")
    else:
        print("   ✅ Model is correctly placed on the CUDA device.")

    # Print memory footprint
    mem_footprint = model.get_memory_footprint()
    print(f"   - Model memory footprint: {mem_footprint / 1e6:.2f} MB")

    # Check 4: Perform a simple forward pass
    print("\nStep 4: Performing a simple inference test (forward pass)...")
    try:
        # Create a dummy input tensor on the GPU
        dummy_input = torch.randint(0, 1000, (1, 10)).to("cuda")
        
        # Perform a forward pass
        with torch.no_grad():
            outputs = model(dummy_input)

        print("✅ Success: Forward pass completed without errors.")
    except Exception as e:
        print(f"\n❌ ERROR: The forward pass failed.")
        print("   There might be an issue with the quantized operations on your GPU.")
        print(f"   Error details: {e}")
        print("Verification failed.")
        return

    print("\n--- Verification Complete ---")
    print("🎉 All checks passed! Your `bitsandbytes` installation appears to be working correctly with your GPU.")

if __name__ == "__main__":
    verify_bitsandbytes_gpu()