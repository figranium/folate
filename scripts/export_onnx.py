import argparse
import os
import tempfile
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Export and dynamically quantize Folate to INT8 ONNX.")
    parser.add_argument("--model-dir", default="./folate-base", help="Path to fine-tuned Hugging Face model")
    parser.add_argument("--output-dir", default="./folate-int8", help="Destination directory for INT8 ONNX model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    dummy = tokenizer("click submit", "[TAG] button [TEXT] Submit", return_tensors="pt")

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, "folate_fp32.onnx")
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            fp32_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

        quantize_dynamic(
            model_input=fp32_path,
            model_output=os.path.join(args.output_dir, "model_quantized.onnx"),
            weight_type=QuantType.QUInt8,
        )

    tokenizer.save_pretrained(args.output_dir)
    print(f"Exported quantized INT8 model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
