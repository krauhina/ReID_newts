import torch
from torchvision import transforms
from bot.handlers.utils.deployment_yolo import process_single_image
from bot.handlers.utils.deployment_vit import find_similar_images


async def photo_processing(bot):
    try:
        input_image_path = f'{bot.save_dir}/image.jpg'
        success = await process_single_image(input_image_path, bot.save_dir)

        if success:
            MODEL_PATH = "bot/models/best_model.pth"
            DATABASE_DIR = "bot/crop_dataset"
            QUERY_IMAGE = f'{bot.save_dir}/image_cropped.jpg'
            OUTPUT_DIR = bot.result_dir
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            TRANSFORMS = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            find_similar_images(
                model_path=MODEL_PATH,
                database_dir=DATABASE_DIR,
                query_image_path=QUERY_IMAGE,
                output_dir=OUTPUT_DIR,
                transform=TRANSFORMS,
                device=DEVICE,
                bot=bot
            )
            return True
        return False

    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")

        return False
