from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='logs/230702_1821_training_cornell/epoch_47_iou_0.99',
        visualize=True
    )
    generator.load_model()
    generator.run()
