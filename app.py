#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from dualstylegan import Model

DESCRIPTION = '''# Portrait Style Transfer with [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN)

<img id="overview" alt="overview" src="https://raw.githubusercontent.com/williamyang1991/DualStyleGAN/main/doc_images/overview.jpg" />
'''


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
    return f'<img id="style-image" src="{url}" alt="style image">'


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
    return gr.update(maximum=max_vals[choice])


def update_style_image(style_name: str) -> dict:
    text = get_style_image_markdown_text(style_name)
    return gr.update(value=text)


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Box():
        gr.Markdown('''## Step 1 (Preprocess Input Image)

- Drop an image containing a near-frontal face to the **Input Image**.
- If there are multiple faces in the image, hit the Edit button in the upper right corner and crop the input image beforehand.
- Hit the **Preprocess** button.
- Choose the encoder version. Default is Z+ encoder which has better stylization performance. W+ encoder better reconstructs the input image to preserve more details.
- The final result will be based on this **Reconstructed Face**. So, if the reconstructed image is not satisfactory, you may want to change the input image.
''')
        with gr.Row():
            encoder_type = gr.Radio(label='Encoder Type',
                                    choices=[
                                        'Z+ encoder (better stylization)',
                                        'W+ encoder (better reconstruction)'
                                    ],
                                    value='Z+ encoder (better stylization)')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image',
                                           type='filepath')
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
                instyle = gr.State()

        with gr.Row():
            paths = sorted(pathlib.Path('images').glob('*.jpg'))
            gr.Examples(examples=[[path.as_posix()] for path in paths],
                        inputs=input_image)

    with gr.Box():
        gr.Markdown('''## Step 2 (Select Style Image)

- Select **Style Type**.
- Select **Style Image Index** from the image table below.
''')
        with gr.Row():
            with gr.Column():
                style_type = gr.Radio(label='Style Type',
                                      choices=model.style_types,
                                      value=model.style_types[0])
                text = get_style_image_markdown_text('cartoon')
                style_image = gr.Markdown(value=text)
                style_index = gr.Slider(label='Style Image Index',
                                        minimum=0,
                                        maximum=316,
                                        step=1,
                                        value=26)

        with gr.Row():
            gr.Examples(
                examples=[
                    ['cartoon', 26],
                    ['caricature', 65],
                    ['arcane', 63],
                    ['pixar', 80],
                ],
                inputs=[style_type, style_index],
            )

    with gr.Box():
        gr.Markdown('''## Step 3 (Generate Style Transferred Image)

- Adjust **Structure Weight** and **Color Weight**.
- These are weights for the style image, so the larger the value, the closer the resulting image will be to the style image.
- Tips: For W+ encoder, better way of (Structure Only) is to uncheck (Structure Only) and set Color weight to 0.
- Hit the **Generate** button.
''')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    structure_weight = gr.Slider(label='Structure Weight',
                                                 minimum=0,
                                                 maximum=1,
                                                 step=0.1,
                                                 value=0.6)
                with gr.Row():
                    color_weight = gr.Slider(label='Color Weight',
                                             minimum=0,
                                             maximum=1,
                                             step=0.1,
                                             value=1)
                with gr.Row():
                    structure_only = gr.Checkbox(label='Structure Only',
                                                 value=False)
                with gr.Row():
                    generate_button = gr.Button('Generate')

            with gr.Column():
                result = gr.Image(label='Result')

        with gr.Row():
            gr.Examples(
                examples=[
                    [0.6, 1.0],
                    [0.3, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                inputs=[structure_weight, color_weight],
            )

    preprocess_button.click(
        fn=model.detect_and_align_face,
        inputs=[input_image],
        outputs=aligned_face,
    )
    aligned_face.change(
        fn=model.reconstruct_face,
        inputs=[aligned_face, encoder_type],
        outputs=[
            reconstructed_face,
            instyle,
        ],
    )
    style_type.change(
        fn=update_slider,
        inputs=style_type,
        outputs=style_index,
    )
    style_type.change(
        fn=update_style_image,
        inputs=style_type,
        outputs=style_image,
    )
    generate_button.click(
        fn=model.generate,
        inputs=[
            style_type,
            style_index,
            structure_weight,
            color_weight,
            structure_only,
            instyle,
        ],
        outputs=result,
    )
demo.queue(max_size=20).launch()
