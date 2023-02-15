from manim import *
import os
import pandas as pd

class part1(Scene):
    def construct(self):
        text1 = Tex(r'\textbf{Vocal Tract Segmentation \\ For Neurodegenerative Disease Applications}', font_size=55).move_to(3*UP)
        text2 = Tex('Group 3').next_to(text1, 2*DOWN)
        name1 = Tex('name1').move_to(2*DOWN + 6*LEFT)
        name2 = Tex('name2').next_to(name1, RIGHT)
        name3 = Tex('name3').next_to(name2, RIGHT)
        name4 = Tex('name4').next_to(name3, RIGHT)
        name5 = Tex('name5').next_to(name4, RIGHT)

        self.play(Write(text1))
        self.play(Write(text2))
        self.play(Write(name1), Write(name2), Write(name3), Write(name4), Write(name5))
        self.wait(3)


class timelapse(Scene):
    def construct(self):
        folder_path_original = '/home/mattia/Projects/neuro_visual/GS_images/s000001'
        folder_path_segmented = '/home/mattia/Projects/neuro_visual/segmented'

        files_original = sorted(os.listdir(folder_path_original))
        files_original = files_original[0:100]

        files_segmented = sorted(os.listdir(folder_path_segmented))
        files_segmented = files_segmented[0:100]

        for file_original, file_segmented in zip(files_original, files_segmented):
            image_path_original = os.path.join(folder_path_original, file_original)
            image_original = ImageMobject(image_path_original).scale(2).move_to(2*UP)

            image_path_segmented = os.path.join(folder_path_segmented, file_segmented)
            image_segmented = ImageMobject(image_path_segmented).scale(2).next_to(image_original, DOWN)

            self.play(FadeIn(image_original), FadeIn(image_segmented), run_time=0.1)

class original_timelapse(Scene):
    def construct(self):
        folder_path_original = '/home/mattia/Projects/neuro_visual/GS_images/s000001'
        
        files_original = sorted(os.listdir(folder_path_original))
        files_original = files_original[0:100]

        for file_original in files_original:
            image_path_original = os.path.join(folder_path_original, file_original)
            image_original = ImageMobject(image_path_original)
            image_original.scale(2.5)

            self.play(FadeIn(image_original), run_time=0.1)

class segmented_timelapse(Scene):
    def construct(self):
        folder_path_original = '/home/mattia/Projects/neuro_visual/segmented'
        
        files_original = sorted(os.listdir(folder_path_original))
        files_original = files_original[0:100]

        for file_original in files_original:
            image_path_original = os.path.join(folder_path_original, file_original)
            image_original = ImageMobject(image_path_original)
            image_original.scale(2.5)

            self.play(FadeIn(image_original), run_time=0.1)

class split_set(Scene):
    def construct(self):

        self.camera.background_color = WHITE

        n_images = [280, 240, 150, 150]
        tot_images = sum(n_images)
        rel_imgs = [i/tot_images for i in n_images]

        title = Tex(r'\textbf{Train/Val/Test split}').move_to(UP*3.5).set_color(BLACK)

        dataset = Rectangle(width=10,height=1).shift(UP*2).set_color(BLACK)
        dataset_text = Tex('Dataset (820 imgs)').move_to(dataset.get_center()).set_color(BLACK)

        s1 = Rectangle(width=10*rel_imgs[0], height=1).shift(LEFT*3.3).set_color(BLACK)
        text1 = Tex(280).move_to(s1.get_center()).set_color(BLACK)
        text11 = Tex('s1', font_size=35).next_to(s1, 0.8*UP).set_color(BLACK)
        V1 = VGroup(s1, text1, text11)

        s2 = Rectangle(width=10*rel_imgs[1], height=1).next_to(s1,RIGHT,buff=0).set_color(BLACK)
        text2 = Tex(240).move_to(s2.get_center()).set_color(BLACK)
        text22 = Tex('s2', font_size=35).next_to(s2, 0.8*UP).set_color(BLACK)
        V2 = VGroup(s2, text2, text22)

        s4 = Rectangle(width=10*rel_imgs[3], height=1).next_to(s2,RIGHT,buff=0).set_color(BLACK)
        text4 = Tex(150).move_to(s4.get_center()).set_color(BLACK)
        text44 = Tex('s3', font_size=35).next_to(s4, 0.8*UP).set_color(BLACK)
        V4 = VGroup(s4, text4, text44)

        s5 = Rectangle(width=10*rel_imgs[3], height=1).next_to(s4,RIGHT,buff=0).set_color(BLACK)
        text5 = Tex(150).move_to(s5.get_center()).set_color(BLACK)
        text55 = Tex('s4', font_size=35).next_to(s5, 0.8*UP).set_color(BLACK)
        V5 = VGroup(s5, text5, text55)

        self.add(title)
        self.play(Create(dataset), Write(dataset_text))
        self.wait()
        self.play(Create(V1), Create(V2), Create(V4), Create(V5), run_time=2)
        self.wait()

        self.play(
            V1.animate.shift(DOWN + 1.2*LEFT),
            V2.animate.shift(DOWN + 1.2*LEFT),
            V4.animate.shift(DOWN + 0.5*LEFT),
            V5.animate.shift(DOWN + 0.5*RIGHT),
        )
        self.wait()

        V_T = VGroup(V1,V2)
        train = Tex('Train set').next_to(V_T, DOWN).set_color(BLACK)
        val = Tex('Val set').next_to(V4, DOWN).set_color(BLACK)
        test = Tex('Test set').next_to(V5, DOWN).set_color(BLACK)

        self.play(Write(train), Write(val), Write(test))
        self.wait(3)

class data_aug(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Data augmentation}').move_to(UP*3.5).set_color(BLACK)
        self.add(title)

        train_set = Rectangle(width=2.5, height=1).set_color(BLACK).shift(LEFT*5 + 1.5*UP)
        copy_train_set = Rectangle(width=2.5, height=1).set_color(RED_E).shift(LEFT*5 + 1.5*DOWN)

        text1 = Tex('Train Set', font_size=30).set_color(BLACK).move_to(train_set.get_center())
        text2 = Tex(r'Copy of\\Train Set', font_size=30).set_color(RED_E).move_to(copy_train_set.get_center())

        V1 = VGroup(train_set, text1)
        V2 = VGroup(copy_train_set, text2)

        self.play(Create(V1))
        self.play(TransformFromCopy(V1,V2))

        augmented_set = Rectangle(width=2.5, height=1).set_color(RED_E).next_to(copy_train_set,RIGHT*5)
        text3 = Tex(r'Augmented\\Set', font_size=30).set_color(RED_E).move_to(augmented_set.get_center())
        V3 = VGroup(augmented_set, text3)

        arrow = Arrow(start=copy_train_set.get_edge_center(RIGHT), end=augmented_set.get_edge_center(LEFT), color=BLACK)

        self.play(Create(arrow))
        self.play(Create(V3))

        final_pt1 = V1.copy().move_to(RIGHT*3)
        final_pt2 = V3.copy().next_to(final_pt1, RIGHT, buff=0)

        line1 = Line(start=V1.get_edge_center(RIGHT), end=(final_pt1.get_edge_center(UP)+UP)).set_color(BLACK)
        line2 = Line(start=V3.get_edge_center(RIGHT), end=(final_pt2.get_edge_center(DOWN)+DOWN)).set_color(BLACK)

        arrow1 = Arrow(start=line1.get_end(), end=final_pt1.get_edge_center(UP), buff=0).set_color(BLACK)
        arrow2 = Arrow(start=line2.get_end(), end=final_pt2.get_edge_center(DOWN), buff=0).set_color(BLACK)

        self.play(Create(line1), Create(line2))
        self.play(Create(arrow1), Create(arrow2))
        self.play(FadeIn(final_pt1), FadeIn(final_pt2))
        self.wait(2)
        V = VGroup(V1, V2, arrow, final_pt1, final_pt2, arrow1, arrow2, line1, line2)
        self.play(FadeOut(V))
        self.play(V3.animate.move_to(UP*2.4))

        img_rotate_path = ['/home/mattia/Projects/neuro_visual/MRI_1.png', '/home/mattia/Projects/neuro_visual/label_1.png']
        img_scale_path = ['/home/mattia/Projects/neuro_visual/MRI_2.png', '/home/mattia/Projects/neuro_visual/label_2.png']
        img_shift_path = ['/home/mattia/Projects/neuro_visual/MRI_4.png', '/home/mattia/Projects/neuro_visual/label_4.png']
        img_mix_path = ['/home/mattia/Projects/neuro_visual/MRI_3.png', '/home/mattia/Projects/neuro_visual/label_3.png']

        img_rotate = ImageMobject(img_rotate_path[0]).move_to(LEFT*4.5).scale(1.5)
        img_scale = ImageMobject(img_scale_path[0]).move_to(LEFT*1.5).scale(1.5)
        img_shift = ImageMobject(img_shift_path[0]).move_to(RIGHT*1.5).scale(1.5)
        img_mix = ImageMobject(img_mix_path[0]).move_to(RIGHT*4.5).scale(1.5)
        V_img = Group(img_rotate, img_scale, img_shift, img_mix)

        img_rotate_label = ImageMobject(img_rotate_path[1]).next_to(img_rotate, 2*DOWN).scale(1.5)
        img_scale_label = ImageMobject(img_scale_path[1]).next_to(img_scale, 2*DOWN).scale(1.5)
        img_shift_label = ImageMobject(img_shift_path[1]).next_to(img_shift, 2*DOWN).scale(1.5)
        img_mix_label = ImageMobject(img_mix_path[1]).next_to(img_mix, 2*DOWN).scale(1.5)
        V_label = Group(img_rotate_label, img_scale_label, img_shift_label, img_mix_label)


        rotate = Tex('Rotate', font_size=35).next_to(img_rotate, 0.6*UP).set_color(BLACK)
        scale = Tex('Scale', font_size=35).next_to(img_scale, 0.6*UP).set_color(BLACK)
        shift = Tex('Shift', font_size=35).next_to(img_shift, 0.6*UP).set_color(BLACK)
        mix = Tex('Mix', font_size=35).next_to(img_mix, 0.6*UP).set_color(BLACK)
        V_text = VGroup(rotate, scale, shift, mix)

        self.play(FadeIn(V_text), FadeIn(V_img), FadeIn(V_label))
        self.wait(3)


class imunet(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{IMU-NET architecture}').set_color(BLACK)
        title.shift(3.5*UP)
        self.add(title)

        img_all_path = '/home/mattia/Projects/neuro_visual/unet_U.png'
        img_enc_path = '/home/mattia/Projects/neuro_visual/unet_U_enc.png'
        img_bridge_path = '/home/mattia/Projects/neuro_visual/unet_U_bridge.png'
        img_dec_path = '/home/mattia/Projects/neuro_visual/unet_U_dec.png'

        image_all = ImageMobject(img_all_path).scale(0.25)
        image_enc = ImageMobject(img_enc_path).scale(0.25)
        image_bridge = ImageMobject(img_bridge_path).scale(0.25)
        image_dec = ImageMobject(img_dec_path).scale(0.25)

        enc_text = Tex('Encoding branch').next_to(image_enc,LEFT).set_color(BLACK)
        bridge_text = Tex('Bridge').next_to(image_bridge,DOWN).set_color(BLACK)
        dec_text = Tex('Decoding branch').next_to(image_dec,RIGHT).set_color(BLACK)

        self.play(FadeIn(image_all))
        self.wait(1)

        self.play(FadeIn(image_enc), FadeIn(enc_text))
        self.wait(1)

        self.play(FadeOut(image_enc), FadeOut(enc_text), FadeIn(image_bridge), FadeIn(bridge_text))
        self.wait(1)

        self.play(FadeOut(image_bridge), FadeOut(bridge_text), FadeIn(image_dec), FadeIn(dec_text))
        self.wait(1)

        self.play(FadeOut(image_dec), FadeOut(dec_text))
        self.wait(1)

        self.play(image_all.animate.shift(LEFT*3))

        frame = Rectangle(height=0.75, width=1.25).move_to(LEFT*4.4+UP*0.75).set_color(RED_E)
        self.play(Create(frame))
        
        frame_enc = Rectangle(height=3.5, width=3.5).move_to(2.5*RIGHT).set_color(RED_E)
        enc_block = ImageMobject('/home/mattia/Projects/neuro_visual/unet_enc.png').move_to(frame_enc.get_center()).scale(0.3)
        self.play(TransformFromCopy(frame, frame_enc), FadeIn(enc_block))
        self.wait()
        self.play(FadeOut(enc_block), FadeOut(frame_enc))
        self.wait()

        self.play(frame.animate.shift(RIGHT*2.65))
        self.wait()
        frame_dec = Rectangle(height=3.5, width=3.5).move_to(2.5*RIGHT).set_color(RED_E)
        dec_block = ImageMobject('/home/mattia/Projects/neuro_visual/unet_dec.png').move_to(frame_dec.get_center()).scale(0.3)
        self.play(TransformFromCopy(frame, frame_dec), FadeIn(dec_block))
        self.wait(2)

        

class preprocessing(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Pre-Processing}').set_color(BLACK)
        title.shift(3.5*UP)
        self.add(title)

        text1 = Tex('Aim: remove Gaussian Noise', font_size=44).set_color(BLACK).move_to(UP*2.6)
        text2 = Tex('\{Rudin, Osher and Fatemi algorithm + high frequency enhancement\}', font_size=40).set_color(BLACK).next_to(text1, DOWN)

        self.play(Create(text1))
        self.play(Create(text2))
        self.wait()

        noise_path = '/home/mattia/Projects/neuro_visual/noise.png'
        no_noise_path = '/home/mattia/Projects/neuro_visual/no_noise.png'
        diff_path = '/home/mattia/Projects/neuro_visual/diff.png'

        img_noise = ImageMobject(noise_path).shift(LEFT*3+ DOWN).scale(2.3)
        img_no_noise = ImageMobject(no_noise_path).shift(RIGHT*3+ DOWN).scale(2.3)
        img_diff = ImageMobject(diff_path).scale(2.3)

        text_noise = Tex('Original', font_size=30).set_color(BLACK).next_to(img_noise, UP*0.8)
        text_no_noise = Tex('Processed', font_size=30).set_color(BLACK).next_to(img_no_noise, UP*0.8)

        V1 = Group(img_noise, text_noise)
        V2 = Group(img_no_noise, text_no_noise)
        

        self.play(FadeIn(V1), FadeIn(V2))

        minus = Tex('-', font_size=60).set_color(BLACK).move_to((V2.get_center()+V1.get_center())/2)

        self.wait()
        self.play(Create(minus))
        self.play(V1.animate.shift(LEFT*2.1), V2.animate.shift(LEFT*3.4), minus.animate.shift(LEFT*2.8))

        equal = Tex('$=$', font_size=50).set_color(BLACK).move_to(minus.get_center()+4.8*RIGHT)

        img_diff.next_to(img_no_noise, 4.2*RIGHT)
        text_diff = Tex('Difference', font_size=30).set_color(BLACK).next_to(img_diff, UP*0.8)

        V3 = Group(img_diff, text_diff)

        self.play(Create(equal))
        self.play(FadeIn(V3))
        self.wait()

        distribution = pd.read_csv('/home/mattia/Projects/neuro_visual/data.csv', header=None)
        distribution = distribution[0].tolist()
        distribution = [i/max(distribution) for i in distribution]

        x = np.arange(0,1,1/256)

        self.play(FadeOut(V1), FadeOut(V2), FadeOut(minus), FadeOut(equal))
        self.play(V3.animate.shift(LEFT*8))

        ax = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=5,
            y_length=3.5,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": True, "tick_size": 0.05, "font_size":24},
            x_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1]},
            y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1]},
        )

        ax.next_to(V3, RIGHT*10).set_color(BLACK)
        

        x_label = ax.get_x_axis_label(
            Tex("$Pixel\,\, intensity$").scale(0.5).set_color(BLACK),
            edge=DOWN,
            direction=DOWN,
            buff=0.4,
        )

        y_label = ax.get_y_axis_label(
            Tex("$Normalized \,\,Histogram$").scale(0.5).rotate(90 * DEGREES).set_color(BLACK),
            edge=LEFT,
            direction=LEFT,
            buff=0.2,
        )

        arrow = Arrow(start=V3.get_edge_center(RIGHT), end=(ax.get_edge_center(LEFT)+LEFT*0.5), stroke_width=4, max_tip_length_to_length_ratio=0.2).set_color(BLACK)
        ax.shift(DOWN*0.2)

        line_graph = ax.plot_line_graph(
            x_values = x,
            y_values = distribution,
            line_color=BLACK,
            stroke_width = 4,
            add_vertex_dots=False
        )
        self.play(Create(arrow))
        self.play(Create(ax), Create(x_label), Create(y_label))
        self.play(Create(line_graph))

        self.wait(3)

        
class ablation(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Ablation}').set_color(BLACK)
        title.shift(3.5*UP)
        self.add(title)

        text = Tex('Hyperparameter space is to big', font_size=44).set_color(BLACK).next_to(title,DOWN*2)
        self.play(Create(text))

        line = Line(start=text.get_edge_center(DOWN), end=(text.get_edge_center(DOWN)+DOWN)).set_color(BLACK)
        line1 = Line(start=line.get_end(), end=(line.get_end()+LEFT*4)).set_color(BLACK)
        line2 = Line(start=line.get_end(), end=(line.get_end()+RIGHT*4)).set_color(BLACK)

        arrow1 = Arrow(start=line1.get_end(), end=(line1.get_end()+1*DOWN), buff=0).set_color(BLACK)
        arrow2 = Arrow(start=line2.get_end(), end=(line2.get_end()+1*DOWN), buff=0).set_color(BLACK)

        self.play(Create(line), Create(line1), Create(line2), run_time=0.5)
        self.play(Create(arrow1), Create(arrow2), run_time=0.5)

        fixed = Tex(r'\textbf{Fixed}', font_size=40).set_color(BLACK).next_to(arrow1, DOWN)
        self.play(Create(fixed))

        hp_fixed = Tex(r'Preprocessing\\Augmentation\\Num filters\\Kernel size\\Unet depth\\Class weights', font_size=35).set_color(BLACK)
        hp_fixed.next_to(fixed,DOWN)
        self.play(Create(hp_fixed))
        self.wait()

        RS = Tex(r'\textbf{Random Search}', font_size=40).set_color(BLACK).next_to(arrow2, DOWN)
        self.play(Create(RS))

        hp_rs = Tex(r'Dropout rate\\Learning rate\\Batch size', font_size=35).set_color(BLACK)
        hp_rs.next_to(RS,DOWN)
        self.play(Create(hp_rs))
        self.wait()

        self.play(
            FadeOut(hp_rs),FadeOut(RS),
            FadeOut(hp_fixed),FadeOut(fixed),
            FadeOut(arrow1),FadeOut(arrow2),
            FadeOut(line),FadeOut(line1),FadeOut(line2),
            FadeOut(text),)

        conf1 = Tex('Config1', font_size=32).set_color(BLACK).move_to([-6.3,2.3,0])
        conf2 = Tex('Config2', font_size=32).set_color(BLUE_E).next_to(conf1, DOWN)
        dots = Tex(r'$\vdots$', font_size=32).set_color(BLACK).next_to(conf2, DOWN)
        confN = Tex('ConfigN', font_size=32).set_color(RED_E).next_to(dots, DOWN)
        V_confs = VGroup(conf1, conf2, confN)

        self.play(Create(conf1), Create(conf2), Create(dots), Create(confN))

        train = Rectangle(width=1.8, height=1).set_color(BLACK).move_to([-3, 1.5, 0])
        train_text = Tex('Train set', font_size=35).set_color(BLACK).move_to(train.get_center())
        V1 = VGroup(train, train_text)

        val = Rectangle(width=1.8, height=1).set_color(BLACK).move_to([3, 1.5, 0])
        val_text = Tex('Val set', font_size=35).set_color(BLACK).move_to(val.get_center())
        V2 = VGroup(val, val_text)

        self.play(Create(V1), Create(V2))

        V_arrow = VGroup()
        for i in range(3):
            arrow = Arrow(start=V_confs[i].get_edge_center(RIGHT), end=train.get_edge_center(LEFT), stroke_width=3, max_tip_length_to_length_ratio=0.1).set_color(BLACK)
            self.play(Create(arrow), run_time=0.2)

        model1 = Tex('Model1', font_size=32).set_color(BLACK).move_to([-0,2.3,0])
        model2 = Tex('Model2', font_size=32).set_color(BLUE_E).next_to(model1, DOWN)
        dots2 = Tex(r'$\vdots$', font_size=32).set_color(BLACK).next_to(model2, DOWN)
        modelN = Tex('ModelN', font_size=32).set_color(RED_E).next_to(dots2, DOWN)
        V_models = VGroup(model1, model2, modelN)

        for i in range(3):
            arrow = Arrow(start=train.get_edge_center(RIGHT), end=V_models[i].get_edge_center(LEFT), stroke_width=3, max_tip_length_to_length_ratio=0.1).set_color(BLACK)
            if i==2:
                self.play(Create(arrow), Create(V_models[i]),Create(dots2), run_time=0.2)
            else:
                self.play(Create(arrow), Create(V_models[i]), run_time=0.2)

        for i in range(3):
            arrow = Arrow(start=V_models[i].get_edge_center(RIGHT), end=val.get_edge_center(LEFT), stroke_width=3, max_tip_length_to_length_ratio=0.1).set_color(BLACK)
            self.play(Create(arrow), run_time=0.3)

        perf1 = Tex('Performance1', font_size=32).set_color(BLACK).move_to([6,2.3,0])
        perf2 = Tex('Performance2', font_size=32).set_color(BLUE_E).next_to(perf1, DOWN)
        dots3 = Tex(r'$\vdots$', font_size=32).set_color(BLACK).next_to(perf2, DOWN)
        perfN = Tex('PerformanceN', font_size=32).set_color(RED_E).next_to(dots3, DOWN)
        V_perfs= VGroup(perf1, perf2, perfN)

        for i in range(3):
            arrow = Arrow(start=val.get_edge_center(RIGHT), end=V_perfs[i].get_edge_center(LEFT), stroke_width=3, max_tip_length_to_length_ratio=0.15).set_color(BLACK)
            if i==2:
                self.play(Create(arrow), Create(V_perfs[i]),Create(dots3), run_time=0.2)
            else:
                self.play(Create(arrow), Create(V_perfs[i]), run_time=0.2)
        
        V_perfs.add(dots3)
        V_perfs2 = V_perfs.copy()
        self.wait()
        self.play(V_perfs2.animate.move_to([-5,-2,0]))
        graph = Tex('\}', font_size=70).set_color(BLACK).next_to(V_perfs2).scale(2.5)
        self.play(Create(graph), run_time=0.5)

        text2 = Tex(r'Choose the\\best one', font_size=35).set_color(BLACK).next_to(graph,RIGHT)
        test = Rectangle(width=1.8, height=1).set_color(BLACK).move_to([1.5,-2,0])
        test_text = Tex('Test set', font_size=35).set_color(BLACK).move_to(test.get_center())
        V3 = VGroup(test, test_text)

        arrow = Arrow(start=text2.get_edge_center(RIGHT), end=test.get_edge_center(LEFT) ,stroke_width=3, max_tip_length_to_length_ratio=0.15)
        arrow.set_color(BLACK)

        self.play(Create(text2), run_time=1)
        self.play(Create(arrow), run_time=0.5)
        self.play(Create(V3), run_time=1)
        self.wait(0.5)

        text3 = Tex(r'Final\\Performances', font_size=35).set_color(BLACK).move_to([5.5,-2,0])
        arrow2 = Arrow(start=test.get_edge_center(RIGHT), end=text3.get_edge_center(LEFT) ,stroke_width=3, max_tip_length_to_length_ratio=0.15)
        arrow2.set_color(BLACK)
        
        self.play(Create(arrow2), run_time=0.5)
        self.play(Create(text3), run_time=1)

        self.wait(1)












