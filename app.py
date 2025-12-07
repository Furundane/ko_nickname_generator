import gradio as gr
from pathlib import Path

from generator_ko import make_nickname_ko
from generator_en import make_nickname_en

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


# GPT-2 한국어 모델 로드
MODEL_DIR = Path("gpt2-nickname-generator/models") / "gpt2_nickname_generator_ko_4l8h_128embd"
gpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(MODEL_DIR))
gpt2_model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))


# 랜덤 닉네임 생성 함수
def generate_rule_based(language: str, tone: str, count: int) -> str:
    names = []
    for _ in range(int(count)):
        if language == "ko":
            t = None if tone == "" else tone
            names.append(make_nickname_ko(tone=t))
        else:
            names.append(make_nickname_en())
    return "\n".join(names)


def generate_gpt2(count: int, temperature: float = 1.0) -> str:
    inputs = gpt2_tokenizer("<BOS>", return_tensors="pt")
    outputs = gpt2_model.generate(
        **inputs,
        max_length=16,
        do_sample=True,
        temperature=float(temperature),
        top_k=50,
        top_p=0.95,
        num_return_sequences=int(count),
        eos_token_id=gpt2_tokenizer.eos_token_id,
        pad_token_id=gpt2_tokenizer.pad_token_id,
    )

    names = []
    for out in outputs:
        text = gpt2_tokenizer.decode(out, skip_special_tokens=True)
        names.append(text)
    return "\n".join(names)


# Gradio용 통합 함수
def generate_both(language: str, tone: str, count: int, temperature: float):
    rule = generate_rule_based(language, tone, count)

    if language == "ko":
        # norm 이외의 톤일 경우 GPT-2는 동작하지 않고 한국어만 지원한다고 출력
        if tone != "norm":
            gpt2 = "GPT-2 닉네임 생성기는 norm 톤만 지원합니다"
        else:
            gpt2 = generate_gpt2(count, temperature)
    else:
        gpt2 = "(GPT-2 닉네임 생성기는 한국어만 지원합니다.)"

    return rule, gpt2


def main() -> None:
    demo = gr.Interface(
        fn=generate_both,
        inputs=[
            gr.Radio(
                choices=["ko", "en"],
                value="ko",
                label="언어",
            ),
            gr.Radio(
                choices=["norm", "intel", "cute", "strong"],
                value="norm",
                label="톤 (한국어 전용)",
            ),
            gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="한번에 생성할 개수",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.2,
                value=1.0,
                step=0.1,
                label="Temperature (1.1을 초과하는 값은 권장하지 않습니다.)",
            ),
        ],
        outputs=[
            gr.Textbox(label="규칙 기반 닉네임", lines=8),
            gr.Textbox(label="GPT-2 닉네임", lines=8),
        ],
        description="규칙 기반 닉네임 / GPT-2 기반 닉네임을 한 번에 생성합니다.",
    )

    # share=True 이면 터널링 활성화.
    demo.launch(share=True)


if __name__ == "__main__":
    main()

