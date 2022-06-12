#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib

import gradio as gr

from dualstylegan import Model

DESCRIPTION = '''# Portrait Style Transfer with <a href="https://github.com/williamyang1991/DualStyleGAN">DualStyleGAN</a>

<img id="overview" alt="overview" src="https://raw.githubusercontent.com/williamyang1991/DualStyleGAN/main/doc_images/overview.jpg" />
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.dualstylegan" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def get_style_image_url(style_name: str) -> str:
    base_url = 'https://raw.githubusercontent.com/williamyang1991/DualStyleGAN/main/doc_images'
    filenames = {
        'cartoon': 'cartoon_overview.jpg',
        'caricature': 'caricature_overview.jpg',
        'anime': 'anime_overview.jpg',
        'arcane': 'Reconstruction_arcane_overview.jpg',
        'comic': 'Reconstruction_comic_overview.jpg',
        'pixar': 'Reconstruction_pixar_overview.jpg',
        'slamdunk': 'Reconstruction_slamdunk_overview.jpg',
    }
    return f'{base_url}/{filenames[style_name]}'


def get_style_image_markdown_text(style_name: str) -> str:
    url = get_style_image_url(style_name)
    return f'<center><img id="style-image" src="{url}" alt="style image"></center>'


def update_slider(choice: str) -> dict:
    max_vals = {
        'cartoon': 316,
        'caricature': 198,
        'anime': 173,
        'arcane': 99,
        'comic': 100,
        'pixar': 121,
        'slamdunk': 119,
    }
    return gr.Slider.update(maximum=max_vals[choice])


def update_style_image(style_name: str) -> dict:
    text = get_style_image_markdown_text(style_name)
    return gr.Markdown.update(value=text)


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_styles(example: list) -> list[dict]:
    return [
        gr.Radio.update(value=example[0]),
        gr.Slider.update(value=example[1]),
    ]


def set_example_weights(example: list) -> list[dict]:
    return [
        gr.Slider.update(value=example[0]),
        gr.Slider.update(value=example[1]),
    ]


def main():
    args = parse_args()
    model = Model(device=args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Box():
            gr.Markdown('''## Step 1 (Preprocess Input Image)

- Drop an image containing a near-frontal face to the **Input Image**.
    - If there are multiple faces in the image, hit the Edit button in the upper right corner and crop the input image beforehand.
- Hit the **Preprocess** button.
    - The final result will be based on this **Reconstructed Face**. So, if the reconstructed image is not satisfactory, you may want to change the input image.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type='file')
                    with gr.Row():
                        preprocess_button = gr.Button('Preprocess')
                with gr.Column():
                    with gr.Row():
                        aligned_face = gr.Image(label='Aligned Face',
                                                type='numpy',
                                                interactive=False)
                with gr.Column():
                    reconstructed_face = gr.Image(label='Reconstructed Face',
                                                  type='numpy')
                    instyle = gr.Variable()

            with gr.Row():
                paths = sorted(pathlib.Path('images').glob('*.jpg'))
                example_images = gr.Dataset(components=[input_image],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 2 (Select Style Image)

- Select **Style Type**.
- Select **Style Image Index** from the image table below.
''')
            with gr.Row():
                with gr.Column():
                    style_type = gr.Radio(model.style_types,
                                          label='Style Type')
                    text = get_style_image_markdown_text('cartoon')
                    style_image = gr.Markdown(value=text)
                    style_index = gr.Slider(0,
                                            316,
                                            value=26,
                                            step=1,
                                            label='Style Image Index')

            with gr.Row():
                example_styles = gr.Dataset(
                    components=[style_type, style_index],
                    samples=[
                        ['cartoon', 26],
                        ['caricature', 65],
                        ['arcane', 63],
                        ['pixar', 80],
                    ])

        with gr.Box():
            gr.Markdown('''## Step 3 (Generate Style Transferred Image)

- Adjust **Structure Weight** and **Color Weight**.
    - These are weights for the style image, so the larger the value, the closer the resulting image will be to the style image.
- Hit the **Generate** button.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        structure_weight = gr.Slider(0,
                                                     1,
                                                     value=0.6,
                                                     step=0.1,
                                                     label='Structure Weight')
                    with gr.Row():
                        color_weight = gr.Slider(0,
                                                 1,
                                                 value=1,
                                                 step=0.1,
                                                 label='Color Weight')
                    with gr.Row():
                        structure_only = gr.Checkbox(label='Structure Only')
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Image(label='Result')

            with gr.Row():
                example_weights = gr.Dataset(
                    components=[structure_weight, color_weight],
                    samples=[
                        [0.6, 1.0],
                        [0.3, 1.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                    ])

        gr.Markdown(FOOTER)

        preprocess_button.click(fn=model.detect_and_align_face,
                                inputs=input_image,
                                outputs=aligned_face)
        aligned_face.change(fn=model.reconstruct_face,
                            inputs=aligned_face,
                            outputs=[
                                reconstructed_face,
                                instyle,
                            ])
        style_type.change(fn=update_slider,
                          inputs=style_type,
                          outputs=style_index)
        style_type.change(fn=update_style_image,
                          inputs=style_type,
                          outputs=style_image)
        generate_button.click(fn=model.generate,
                              inputs=[
                                  style_type,
                                  style_index,
                                  structure_weight,
                                  color_weight,
                                  structure_only,
                                  instyle,
                              ],
                              outputs=result)
        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)
        example_styles.click(fn=set_example_styles,
                             inputs=example_styles,
                             outputs=example_styles.components)
        example_weights.click(fn=set_example_weights,
                              inputs=example_weights,
                              outputs=example_weights.components)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
