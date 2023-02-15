from manim import *
import os
import pickle
     


class EpochsProgression(Scene):
    def construct(self):

        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Results}').move_to(UP*3.5).set_color(BLACK)
        self.add(title)

        config_1 = Tex('Initial block: 16 filters', font_size=28)
        config_1.set_color(BLACK)
        config_1.next_to(title, 2*DOWN+LEFT)

        config_2 = Tex('Encoding path: [32,64,128,256] filters', font_size=28)
        config_2.set_color(BLACK)
        config_2.next_to(config_1, DOWN)

        config_3 = Tex('Bridge: [512,512,512,512] filters', font_size=28)
        config_3.set_color(BLACK)
        config_3.next_to(config_2, DOWN)

        config_4 = Tex('Decoding path: [256,128,64,32] filters', font_size=28)
        config_4.set_color(BLACK)
        config_4.next_to(config_3, DOWN)

        config_5 = Tex('Learning rate: $1e^{-3}$ ', font_size=28)
        config_5.set_color(BLACK)
        config_5.next_to(config_1, RIGHT*6.5)

        config_6 = Tex('Dropout rate: 0', font_size=28)
        config_6.set_color(BLACK)
        config_6.next_to(config_5, DOWN)

        config_7 = Tex('Batch size: 8', font_size=28)
        config_7.set_color(BLACK)
        config_7.next_to(config_6, DOWN)

        config_8 = Tex(r'Squeeze \& Excitation', font_size=28)
        config_8.set_color(BLACK)
        config_8.next_to(config_7, DOWN*1.2)

        V1 = VGroup(config_1,config_2,config_3,config_4,
                    config_5,config_6,config_7,config_8)
        V1.shift(DOWN*0.7+ RIGHT*3)
        


        epoch_text = Tex('Epoch:', font_size = 40).set_color(BLACK)
        epoch = Integer(font_size=40).next_to(epoch_text, RIGHT).set_color(BLACK)

        V0 = Group(epoch, epoch_text).next_to(V1, UP)

        num_epochs = len(os.listdir('./Predictions Images'))

        with open('trainHistoryDict', 'rb') as f:
            history = pickle.load(f)

        loss = history['val_loss']
        dice = history['val_Mean_DICE']
        recall = history['val_recall_1']
        precision = history['val_precision_1']
        hau = history['val_Hausdorff']


        ax_dice = Axes(
            x_range=[0, num_epochs, 2],
            y_range=[0, 1, 0.2],
            x_length=3.5,
            y_length=2.5,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": True, "tick_size": 0.05, "font_size":24},
            x_axis_config={"numbers_to_include": np.arange(0, num_epochs, 10)},
            y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1]},
        )

        ax_dice.move_to([-4.5, -2, 0]).set_color(BLACK)

        dice_x_label = ax_dice.get_x_axis_label(
            Tex("epoch").scale(0.5).set_color(BLACK),
            edge=DOWN,
            direction=DOWN,
            buff=0.4,
        )

        dice_y_label = ax_dice.get_y_axis_label(
            Tex("DICE").scale(0.5).rotate(90 * DEGREES).set_color(BLACK),
            edge=LEFT,
            direction=LEFT,
            buff=0.2,
        )

        V2 = Group(ax_dice, dice_x_label, dice_y_label)

        ax_recall = Axes(
            x_range=[0, num_epochs, 2],
            y_range=[0, 1, 0.2],
            x_length=3.5,
            y_length=2.5,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": True, "tick_size": 0.05, "font_size":24},
            x_axis_config={"numbers_to_include": np.arange(0, num_epochs, 10)},
            y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1]},
        )

        ax_recall.next_to(ax_dice,RIGHT).set_color(BLACK)

        recall_x_label = ax_recall.get_x_axis_label(
            Tex("epoch").scale(0.5).set_color(BLACK),
            edge=DOWN,
            direction=DOWN,
            buff=0.4,
        )

        recall_y_label = ax_recall.get_y_axis_label(
            Tex("Recall").scale(0.5).rotate(90 * DEGREES).set_color(BLACK),
            edge=LEFT,
            direction=LEFT,
            buff=0.2,
        )

        V3 = Group(ax_recall, recall_x_label, recall_y_label)

        ax_precision = Axes(
            x_range=[0, num_epochs, 2],
            y_range=[0, 1, 0.2],
            x_length=3.5,
            y_length=2.5,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": True, "tick_size": 0.05, "font_size":24},
            x_axis_config={"numbers_to_include": np.arange(0, num_epochs, 10)},
            y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1]},
        )

        ax_precision.next_to(ax_recall, RIGHT).set_color(BLACK)

        precision_x_label = ax_precision.get_x_axis_label(
            Tex("epoch").scale(0.5).set_color(BLACK),
            edge=DOWN,
            direction=DOWN,
            buff=0.4,
        )

        precision_y_label = ax_precision.get_y_axis_label(
            Tex("Precision").scale(0.5).rotate(90 * DEGREES).set_color(BLACK),
            edge=LEFT,
            direction=LEFT,
            buff=0.2,
        )

        V4 = Group(ax_precision, precision_x_label, precision_y_label)

        self.play(FadeIn(V1, V0, V2, V3, V4))

        dots_dice = []
        dots_recall = []
        dots_precision = []
        dots_hau = []

        for i in range(num_epochs):
            dot_dice = Dot(ax_dice.coords_to_point(i,dice[i]), radius=0.04, color=BLUE_E)
            dot_recall = Dot(ax_recall.coords_to_point(i,recall[i]), radius=0.04, color=BLUE_E)
            dot_precision = Dot(ax_precision.coords_to_point(i,precision[i]), radius=0.04, color=BLUE_E)


            dots_dice.append(dot_dice)
            dots_recall.append(dot_recall)
            dots_precision.append(dot_precision)

        images = []

        for i in range(num_epochs):
            image = ImageMobject('/home/mattia/Projects/neuro_visual/Predictions Images/' + str(i) + '.png')
            image.scale(2.3)
            image.shift(LEFT*4.5+UP*1.5)

            images.append(image)

        for i in range(num_epochs):
            
            if i>0:
                dice_current = dots_dice[i]
                dice_previous = dots_dice[i-1]
                dice_line = Line(dice_previous.get_center(), dice_current.get_center()).set_color(BLUE_E)

                recall_current = dots_recall[i]
                recall_previous = dots_recall[i-1]
                recall_line = Line(recall_previous.get_center(), recall_current.get_center()).set_color(BLUE_E)

                precision_current = dots_precision[i]
                precision_previous = dots_precision[i-1]
                precision_line = Line(precision_previous.get_center(), precision_current.get_center()).set_color(BLUE_E)


                self.play(FadeIn(images[i]), epoch.animate.set_value(i+1),
                          Create(dice_line), Create(recall_line), Create(precision_line),run_time=0.15)
                self.remove(images[i-1])
            else:
                self.play(FadeIn(images[0]), epoch.animate.set_value(i+1),
                          Create(dots_dice[i]), Create(dots_recall[i]), Create(dots_precision[i]), run_time=0.15)

        self.play(FadeOut(V1), FadeOut(V0))

        base = Tex(r'\textbf{Baseline}', font_size=30).set_color(BLACK).next_to(V0, 0.5*DOWN)
        base_results = Tex('DICE: 0.79 \quad Recall: 0.97 \quad Precision: 0.97', font_size = 30).next_to(base,DOWN).set_color(BLACK)

        impr = Tex(r'\textbf{Improved}', font_size=30).set_color(BLACK).next_to(base_results, DOWN)
        impr_results = Tex('DICE: 0.90 \quad Recall: 0.98 \quad Precision: 0.98', font_size = 30).next_to(impr,DOWN).set_color(BLACK)

        self.play(Create(base), Create(base_results), Create(impr), Create(impr_results))

        self.wait(3)

class results(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Results}').move_to(UP*3.5).set_color(BLACK)
        self.add(title)


        image0 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_0.png').move_to([-6,2,0])
        label0 = Tex('Background', font_size=30).set_color(BLACK).next_to(image0,DOWN)
        self.play(FadeIn(image0), Create(label0), run_time = 0.4)

        image1 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_1.png').next_to(image0, RIGHT*1.5)
        label1 = Tex('Upper lip', font_size=30).set_color(BLACK).next_to(image1,DOWN)
        self.play(FadeIn(image1), Create(label1), run_time = 0.4)

        image2 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_2.png').next_to(image1, RIGHT*1.5)
        label2 = Tex('Hard palate', font_size=30).set_color(BLACK).next_to(image2,DOWN)
        self.play(FadeIn(image2), Create(label2), run_time = 0.4)

        image3 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_3.png').next_to(image2, RIGHT*1.5)
        label3 = Tex('Soft palate', font_size=30).set_color(BLACK).next_to(image3, DOWN)
        self.play(FadeIn(image3), Create(label3), run_time = 0.4)

        image4 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_4.png').next_to(image3, RIGHT*1.5)
        label4 = Tex('Tongue', font_size=30).set_color(BLACK).next_to(image4, DOWN)
        self.play(FadeIn(image4), Create(label4), run_time = 0.4)

        image5 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_5.png').next_to(image4, RIGHT*1.5)
        label5 = Tex('Lower lip', font_size=30).set_color(BLACK).next_to(image5, DOWN)
        self.play(FadeIn(image5), Create(label5), run_time = 0.4)

        image6 = ImageMobject('/home/mattia/Projects/neuro_visual/tttt/label_6.png').next_to(image5, RIGHT*1.5)
        label6 = Tex('Head', font_size=30).set_color(BLACK).next_to(image6, DOWN)
        self.play(FadeIn(image6), Create(label6), run_time = 0.4)

        base_text = Tex(r'\textbf{Baseline: DICE}', font_size=38).set_color(BLACK)

        b0 = Tex('0.98', font_size=37).set_color(BLACK).next_to(label0, DOWN*5)
        b1 = Tex('0.71', font_size=37).set_color(BLACK).next_to(label1, DOWN*5)
        b2 = Tex('0.61', font_size=37).set_color(BLACK).next_to(label2, DOWN*5)
        b3 = Tex('0.56', font_size=37).set_color(BLACK).next_to(label3, DOWN*5)
        b4 = Tex('0.88', font_size=37).set_color(BLACK).next_to(label4, DOWN*5)
        b5 = Tex('0.84', font_size=37).set_color(BLACK).next_to(label5, DOWN*5)
        b6 = Tex('0.94', font_size=37).set_color(BLACK).next_to(label6, DOWN*5)

        V1 = VGroup(b0, b1, b2, b3, b4, b5, b6)
        self.play(Create(base_text), run_time=0.5)
        self.play(Create(V1))
        self.wait()

        f_text = Tex(r'\textbf{Improved: DICE}', font_size=38).set_color(BLACK).next_to(base_text, DOWN*6)

        f0 = Tex('0.99', font_size=37).set_color(BLACK).next_to(b0, DOWN*6)
        f1 = Tex('0.90', font_size=37).set_color(BLACK).next_to(b1, DOWN*6)
        f2 = Tex('0.82', font_size=37).set_color(BLACK).next_to(b2, DOWN*6)
        f3 = Tex('0.80', font_size=37).set_color(BLACK).next_to(b3, DOWN*6)
        f4 = Tex('0.93', font_size=37).set_color(BLACK).next_to(b4, DOWN*6)
        f5 = Tex('0.90', font_size=37).set_color(BLACK).next_to(b5, DOWN*6)
        f6 = Tex('0.97', font_size=37).set_color(BLACK).next_to(b6, DOWN*6)

        V2 = VGroup(f0, f1, f2, f3, f4, f5, f6)
        self.play(Create(f_text), run_time=0.5)
        self.play(Create(V2))

        i0 = Tex(r'$(\uparrow 1\%)$', font_size=26).set_color(BLACK).next_to(f0, DOWN)
        i1 = Tex(r'$(\uparrow 27\%)$', font_size=26).set_color(BLACK).next_to(f1, DOWN)
        i2 = Tex(r'$(\uparrow 34\%)$', font_size=26).set_color(BLACK).next_to(f2, DOWN)
        i3 = Tex(r'$(\uparrow 43\%)$', font_size=26).set_color(BLACK).next_to(f3, DOWN)
        i4 = Tex(r'$(\uparrow 6\%)$', font_size=26).set_color(BLACK).next_to(f4, DOWN)
        i5 = Tex(r'$(\uparrow 7\%)$', font_size=26).set_color(BLACK).next_to(f5, DOWN)
        i6 = Tex(r'$(\uparrow 3\%)$', font_size=26).set_color(BLACK).next_to(f6, DOWN)

        V3 = VGroup(i0, i1, i2, i3, i4, i5, i6)
        self.add(V3)


        self.wait(2)


class conclusions(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        title = Tex(r'\textbf{Conclusions}').move_to(UP*3.5).set_color(BLACK)
        self.add(title)

        img_all_path = '/home/mattia/Projects/neuro_visual/unet_U.png'
        
        image_all = ImageMobject(img_all_path).scale(0.25)
  
        self.play(FadeIn(image_all))
        self.play(image_all.animate.shift(LEFT*3))

        strr = r'''\begin{itemize} 
                     \item Preprocessing for Gaussian noise
                     \item Ablation on IMU-NET
                     \item DICE: 0.90
                     \item Recall: 0.98
                     \item Precision: 0.98
                     \end{itemize} '''

        bullets = Tex(strr, font_size=38).set_color(BLACK).move_to(3*RIGHT)
        self.play(Create(bullets), run_time = 3)
        self.wait(3)
