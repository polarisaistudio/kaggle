!pip install -U transformers

from transformers import pipeline
import torch

# Install Hugging Face CLI if needed
!pip install huggingface_hub

# Login to Hugging Face
!huggingface-cli login


from transformers import AutoProcessor, AutoModelForImageTextToText
import time
import torch

# Use Hugging Face instead of Kaggle
model_name = "google/gemma-3n-e2b"

print("📥 Loading Gemma 3n from Hugging Face...")
print("⚠️  Note: You may need to accept the license on Hugging Face first")
print("   Visit: https://huggingface.co/google/gemma-3n-e2b")

try:
    # Load processor and model from Hugging Face
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print("✅ Model loaded successfully!")
    print("🎓 Starting Polaris AI Educational Testing\n")

    # Educational test prompts
    educational_tests = [
        {
            "age_group": "3-5 years",
            "prompt": "Create a simple story about counting from 1 to 5 using animals that a preschooler would love.",
            "focus": "Early counting skills"
        },
        {
            "age_group": "6-8 years", 
            "prompt": "Explain why leaves change color in fall, using simple words a first grader can understand.",
            "focus": "Nature science"
        },
        {
            "age_group": "9-11 years",
            "prompt": "Create a fun math word problem about sharing pizza that teaches fractions to a 4th grader.",
            "focus": "Math application"
        },
        {
            "age_group": "General",
            "prompt": "You are a patient AI teacher. A child is frustrated with homework. How would you encourage them?",
            "focus": "Emotional support"
        }
    ]

    def generate_educational_response(prompt):
        """Generate response and measure performance"""
        start_time = time.time()
        
        # Process the prompt
        input_ids = processor(text=prompt, return_tensors="pt").to(model.device, dtype=model.dtype)
        
        # Generate response
        outputs = model.generate(
            **input_ids, 
            max_new_tokens=256,
            disable_compile=True,
            temperature=0.7,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        # Decode response
        text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return text[0], response_time

    def evaluate_response(prompt, response, age_group, focus):
        """Simple evaluation framework"""
        print(f"🎯 Age Group: {age_group}")
        print(f"📚 Focus Area: {focus}")
        print(f"❓ Prompt: {prompt}")
        print(f"🤖 AI Response: {response}")
        print("\n📋 Evaluation Questions:")
        print("  [ ] Is the language appropriate for the age group?")
        print("  [ ] Is the content educational and engaging?")
        print("  [ ] Is it safe and appropriate for children?")
        print("  [ ] Would this help a child learn?")
        print("  [ ] Is the response encouraging and positive?")
        print("=" * 80)

    # Run educational tests
    for i, test in enumerate(educational_tests, 1):
        print(f"\n🧪 TEST {i}/{len(educational_tests)}: {test['age_group']} - {test['focus']}")
        print("⏳ Generating response...")
        
        try:
            response, response_time = generate_educational_response(test['prompt'])
            
            # Clean up response (remove the original prompt if it's repeated)
            if test['prompt'] in response:
                response = response.replace(test['prompt'], "").strip()
            
            print(f"⚡ Response Time: {response_time:.2f} seconds")
            evaluate_response(test['prompt'], response, test['age_group'], test['focus'])
            
            # Wait a moment between tests
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            continue

    # Interactive testing session
    print("\n🎮 INTERACTIVE TESTING SESSION")
    print("Now you can test your own educational prompts!")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_prompt = input("🎓 Enter an educational prompt (or 'quit'): ")
            
            if user_prompt.lower() == 'quit':
                break
                
            if len(user_prompt.strip()) == 0:
                continue
                
            print("⏳ Generating response...")
            response, response_time = generate_educational_response(user_prompt)
            
            # Clean response
            if user_prompt in response:
                response = response.replace(user_prompt, "").strip()
                
            print(f"\n🤖 AI Teacher Response:")
            print(f"{response}")
            print(f"\n⚡ Response Time: {response_time:.2f} seconds")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print("\n🎯 Testing Complete!")
    print("💡 Next Steps:")
    print("  1. Review the responses for educational quality")
    print("  2. Note which age groups work best")  
    print("  3. Test with images if available")
    print("  4. Start building your competition demo!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\n💡 Troubleshooting:")
    print("1. Visit https://huggingface.co/google/gemma-3n-e2b and accept the license")
    print("2. Run: huggingface-cli login")
    print("3. Or try the Kaggle version after setting up kaggle.json")
    print("4. Make sure you have enough RAM (model needs ~8GB)")
    
