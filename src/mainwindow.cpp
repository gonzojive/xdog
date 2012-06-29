//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
#include "mainwindow.h"
#include "cudadevicedialog.h"
#include "imageutil.h"
#include "gpu_image.h"
#include "gpu_color.h"
#include "gpu_st.h"
#include "gpu_stgauss2.h"
#include "gpu_stbf2.h"
#include "gpu_gauss.h"
#include "gpu_util.h"
#include "gpu_st.h"
#include "gpu_wog.h"
#include "gpu_bilateral.h"
#include "gpu_dog.h"
#include "gpu_dog2.h"
#include "gpu_etf.h"
#include "gpu_noise.h"
#include "gpu_oabf.h"
#include "gpu_blend.h"
#include "gpu_shuffle.h"
#include "version.h"


//#define ANGLE_PHI 1


MainWindow::MainWindow() {
    setupUi(this);
    m_dirty = false;

    m_logWindow->setVisible(false);
    m_imageView->setFocus();
    m_imageView->setHandler(this);

    ParamGroup *g, *gg;

    new ParamChoice(this, "output", "edges", "edges|fill|fill+edges", &output);
    new ParamChoice(this, "input_gamma", "linear-rgb", "srgb|linear-rgb", &input_gamma);

    g = new ParamGroup(this, "structure_tensor");
    new ParamChoice(g, "st_type", "scharr-lab", "central-diff|sobel-rgb|sobel-lab|sobel-L|scharr-rgb|scharr-lab|gaussian-deriv|etf-full|etf-xy", &st_type);
    new ParamDouble(g, "sigma_c", 2.28, 0, 20, 0.1, &sigma_c);
    new ParamDouble(g, "precision_sigma_c", sqrt(-2*log(0.05)), 1, 10, 1, &precision_sigma_c);
    new ParamInt   (g, "etf_N", 3, 0, 10, 1, &etf_N);

    g = new ParamGroup(this, "bilateral_filter", false, &enable_bf);
    new ParamChoice(g, "type", "xy", "oa|xy|fbl|full", &filter_type);
    new ParamInt   (g, "n_e",     1, 0, 20, 1, &n_e);
    new ParamInt   (g, "n_a",     4, 0, 20, 1, &n_a);
    new ParamDouble(g, "sigma_dg", 3, 0, 20, 0.05, &sigma_dg);
    new ParamDouble(g, "sigma_dt", 3, 0, 20, 0.05, &sigma_dt);
    new ParamDouble(g, "sigma_rg", 4.25, 0, 100, 0.05, &sigma_rg);
    new ParamDouble(g, "sigma_rt", 4.25, 0, 100, 0.05, &sigma_rt);
    new ParamDouble(g, "bf_alpha", 0, 0, 10000, 1, &bf_alpha);
    new ParamDouble(g, "precision_g", 2, 1, 10, 1, &precision_g);
    new ParamDouble(g, "precision_t", 2, 1, 10, 1, &precision_t);

    g = new ParamGroup(this, "dog");
    ParamGroup* dog_group = g;
    connect(g, SIGNAL(dirty()), SLOT(dogChanged()));

    new ParamChoice(g, "type", "flow-based", "isotropic|flow-based", &dog_type);
    new ParamDouble(g, "sigma_e", 1.4, 0, 20, 0.005, &sigma_e);
    new ParamDouble(g, "dog_k", 1.6, 1, 10, 0.01, &dog_k);
    new ParamDouble(g, "precision_e", 3, 1, 5, 0.1, &precision_e);
    new ParamDouble(g, "sigma_m", 4.4, 0, 20, 1, &sigma_m);
    new ParamDouble(g, "precision_m", 2, 1, 5, 0.1, &precision_m);
    new ParamDouble(g, "step_m", 1, 0.01, 2, 0.1, &step_m);

    new ParamChoice(g, "adj_func", "smoothstep", "smoothstep|tanh", &dog_adj_func);
    new ParamBool  (g, "dog_reparam", true, &dog_reparam);
    gg = new ParamGroup(g, "", true);
    dog_tau_g = gg;
    dog_eps_ptr = new ParamDouble(gg, "epsilon", 3.50220, -100, 100, 0.005, &dog_eps);
    dog_tau_ptr = new ParamDouble(gg, "tau", 0.95595, 0, 2, 0.005, &dog_tau);
    dog_phi_ptr = new ParamDouble(gg, "phi", 0.3859, 0, 1e32, 0.1, &dog_phi);

    gg = new ParamGroup(g, "", false);
    dog_p_g = gg;
    dog_p_ptr     = new ParamDouble(gg, "p", 21.7, 0, 1e6, 1, &dog_p);
    dog_eps_p_ptr = new ParamDouble(gg, "epsilon_p", 79.5, -1e32, 1e32, 0.5, &dog_eps_p);
    dog_phi_p_ptr = new ParamDouble(gg, "phi_p", 0.017, -1e32, 1e32, 0.05, &dog_phi_p);

    new ParamChoice(g, "dog_fgauss", "euler", "euler|rk2-nn|rk2|rk4", &dog_fgauss);

    g = new ParamGroup(this, "quantization", false, &quantization);
    new ParamChoice(g, "quant_type", "adaptive", "fixed|adaptive", &quant_type);
    new ParamInt   (g, "nbins", 8, 1, 255, 1, &nbins);
    new ParamDouble(g, "phi_q", 2, 0, 100, 0.025, &phi_q);
    new ParamDouble(g, "lambda_delta", 0, 0, 100, 1, &lambda_delta);
    new ParamDouble(g, "omega_delta", 2, 0, 100, 1, &omega_delta);
    new ParamDouble(g, "lambda_phi", 0.9, 0, 100, 1, &lambda_phi);
    new ParamDouble(g, "omega_phi", 1.6, 0, 100, 1, &omega_phi);

    g = new ParamGroup(this, "warp_sharp", false, &warp_sharp);
    new ParamDouble(g, "sigma_w", 1.5, 0, 20, 1, &sigma_w);
    new ParamDouble(g, "precision_w", 2, 1, 5, 0.1, &precision_w);
    new ParamDouble(g, "phi_w", 2.7, 0, 100, 0.025, &phi_w);

    g = new ParamGroup(this, "final_smooth", true, &final_smooth);
    new ParamChoice(g, "type", "flow-nearest", "3x3|5x5|flow-nearest|flow-linear", &final_type);
    new ParamDouble(g, "sigma_a", 1.0, 0, 10, 1, &sigma_a);

    QScrollArea *sa = new QScrollArea(this);
    QWidget *parea = new QWidget(sa);
    sa->setSizePolicy(QSizePolicy::Fixed,QSizePolicy::Expanding);
    sa->setFixedWidth(300);
    sa->setWidget(parea);
    sa->setFrameStyle(QFrame::NoFrame);
    sa->setFocusPolicy(Qt::NoFocus);
    sa->setWidgetResizable(true);
    sa->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    m_vbox1->addWidget(sa);

    m_paramui = new ParamUI(parea, this);
    QVBoxLayout *pbox = new QVBoxLayout(parea);
    pbox->setContentsMargins(4,4,4,4);
    pbox->addWidget(m_paramui);
    pbox->addStretch(0);

    connect(m_select, SIGNAL(currentIndexChanged(int)), this, SLOT(onIndexChanged(int)));

    m_player = new VideoPlayer(this, ":/test.png");
    connect(m_player, SIGNAL(videoChanged(int)), this, SLOT(onVideoChanged(int)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), this, SLOT(setDirty()));
    connect(m_player, SIGNAL(outputChanged(const QImage&)), m_imageView, SLOT(setImage(const QImage&)));
    connect(this, SIGNAL(imageChanged(const QImage&)), m_player, SLOT(setOutput(const QImage&)));

    m_videoControls->setFrameStyle(QFrame::NoFrame);
    m_videoControls->setAutoHide(true);
    connect(m_videoControls, SIGNAL(stepForward()), m_player, SLOT(stepForward()));
    connect(m_videoControls, SIGNAL(stepBack()), m_player, SLOT(stepBack()));
    connect(m_videoControls, SIGNAL(currentFrameTracked(int)), m_player, SLOT(setCurrentFrame(int)));
    connect(m_videoControls, SIGNAL(playbackChanged(bool)), m_player, SLOT(setPlayback(bool)));
    connect(m_videoControls, SIGNAL(trackingChanged(bool)), this, SLOT(setDirty()));

    connect(m_player, SIGNAL(videoChanged(int)), m_videoControls, SLOT(setFrameCount(int)));
    connect(m_player, SIGNAL(playbackChanged(bool)), m_videoControls, SLOT(setPlayback(bool)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), m_videoControls, SLOT(setCurrentFrame(int)));
}


MainWindow::~MainWindow() {
}


void MainWindow::restoreAppState() {
    QSettings settings;
    restoreGeometry(settings.value("mainWindow/geometry").toByteArray());
    restoreState(settings.value("mainWindow/windowState").toByteArray());

    m_select->setCurrentIndex(settings.value("show_result").toInt());
    actionLog->setChecked(settings.value("show_log", false).toBool());
    settings.beginGroup("imageView");
    m_imageView->restoreSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::restoreSettings(settings, this);
    settings.endGroup();

    settings.beginGroup("paramui");
    ParamUI::restoreSettings(settings, m_paramui);
    settings.endGroup();

    m_player->restoreSettings(settings);
}


void MainWindow::saveAppState() {
    QSettings settings;
    settings.setValue("mainWindow/geometry", saveGeometry());
    settings.setValue("mainWindow/windowState", saveState());

    settings.setValue("show_result", m_select->currentIndex());
    settings.setValue("show_log", actionLog->isChecked());
    settings.beginGroup("imageView");
    m_imageView->saveSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::saveSettings(settings, this);
    settings.endGroup();

    settings.beginGroup("paramui");
    ParamUI::saveSettings(settings, m_paramui);
    settings.endGroup();

    m_player->saveSettings(settings);
}


bool MainWindow::event(QEvent *event) {
    if (event->type() == QEvent::Close) {
        saveAppState();
    }
    bool result = QMainWindow::event(event);
    if (event->type() == QEvent::Polish) {
        restoreAppState();
    }
    return result;
}


void MainWindow::on_actionOpen_triggered() {
    m_player->open();
}


void MainWindow::on_actionAbout_triggered() {
    QMessageBox msgBox;
    msgBox.setWindowTitle("About");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(
        "<html><body>" \
        "<p>Copyright (C) 2010-2012 Hasso-Plattner-Institut,<br/>" \
        "Fachgebiet Computergrafische Systeme &lt;" \
        "<a href='http://www.hpi3d.de'>www.hpi3d.de</a>&gt;<br/><br/>" \
        "Author: Jan Eric Kyprianidis &lt;" \
        "<a href='http://www.kyprianidis.com'>www.kyprianidis.com</a>&gt;<br/>" \
        "Date: " __DATE__ " (" PACKAGE_VERSION ")</p>" \
        "<p>This program is free software: you can redistribute it and/or modify " \
        "it under the terms of the GNU General Public License as published by " \
        "the Free Software Foundation, either version 3 of the License, or " \
        "(at your option) any later version.</p>" \
        "<p>This program is distributed in the hope that it will be useful, " \
        "but WITHOUT ANY WARRANTY; without even the implied warranty of " \
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the " \
        "GNU General Public License for more details.</p>" \
        "Related Publications:" \
        "<ul>" \
        "<li>" \
        "Winnem&ouml;ller, H., Kyprianidis, J. E., &amp; Olsen, S. C. (2012). " \
        "XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization. "\
        "<em>Computers & Graphics</em>, 36(6), pp. 740-753." \
        "</li>" \
        "<li>" \
        "Winnem&ouml;ller, H. (2012). " \
        "XDoG: Advanced image stylization with eXtended Difference-of-Gaussians. "\
        "In: <em>Proc. Symposium on Non-Photorealistic Animation and Rendering (NPAR)</em>, pp. 147-156." \
        "</li>" \
        "<li>" \
        "Kyprianidis, J. E., &amp; D&ouml;llner, J. (2008). " \
        "Image Abstraction by Structure Adaptive Filtering. " \
        "In <em>Proc. EG UK Theory and Practice of Computer Graphics</em>, pp. 51-58." \
        "</li>" \
        "<li>" \
        "Winnem&ouml;ller, H., Olsen, S. C., &amp; Gooch, B. (2012). " \
        "Real-Time Video Abstraction. " \
        "<em>ACM Transactions on Graphics</em>, 25(3), pp. 1221-1226." \
        "</li>" \
        "</ul>" \
        "<p>Test image courtesy of Mickael Casol (CC by 2.0).</p>"
        "</body></html>"
    );
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}


void MainWindow::on_actionSelectDevice_triggered() {
    int current = 0;
    cudaGetDevice(&current);
    int N = CudaDeviceDialog::select(true);
    if ((N >= 0) && (current != N)) {
        QMessageBox::information(this, "Information", "Application must be restarted!");
        qApp->quit();
    }
}

void MainWindow::on_actionRecord_triggered() {
    m_player->record();
}


void MainWindow::setDirty() {
    if (m_videoControls->isTracking()) {
        imageChanged(m_player->image());
    }
    else if (!m_dirty) {
        m_dirty = true;
        QMetaObject::invokeMethod(this, "process", Qt::QueuedConnection);
    }
}


void MainWindow::process() {
    m_dirty = false;
    QImage qsrc = m_player->image();
    if (qsrc.isNull()) {
        m_result[0] = m_result[1] = m_result[2] = qsrc;
        imageChanged(image());
        return;
    }

    gpu_image<float4> src = gpu_image_from_qimage<float4>(qsrc);
    if (input_gamma == "linear-rgb") {
        src = gpu_linear2srgb(src);
    }

    gpu_image<float4> lab = gpu_rgb2lab(src);

    gpu_image<float4> st;
    if (st_type == "central-diff") {
        st = gpu_st_central_diff(gpu_rgb2gray(src));
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else if (st_type == "sobel-rgb") {
        st = gpu_st_sobel(src);
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else if (st_type == "sobel-lab") {
        st = gpu_st_sobel(lab);
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else if (st_type == "sobel-L") {
        st = gpu_st_sobel(gpu_shuffle(lab,0));
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else  if (st_type == "scharr-rgb") {
        st = gpu_st_scharr(src);
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else  if (st_type == "scharr-lab") {
        st = gpu_st_scharr(lab);
        st = gpu_gauss_filter_xy(st, sigma_c, precision_sigma_c);
    }
    else if (st_type == "etf-full") {
        gpu_image<float2> tfm = gpu_etf_full(gpu_rgb2gray(src), sigma_c, etf_N);
        st = gpu_st_from_tangent(tfm);
    }
    else if (st_type == "etf-xy") {
        gpu_image<float2> tfm = gpu_etf_xy(gpu_rgb2gray(src), sigma_c, etf_N);
        st = gpu_st_from_tangent(tfm);
    }
    else if (st_type == "gaussian-deriv") {
        st = gpu_st_gaussian(gpu_rgb2gray(src), sqrtf(0.433f*0.433f + sigma_c*sigma_c));
    }
    else {
        assert(0);
    }

    gpu_image<float4> lfm = gpu_st_lfm(st, bf_alpha);

    gpu_image<float> flow;
    if (!m_noise.is_valid() || (m_noise.size() != src.size())) {
        m_noise = gpu_noise_random(st.w(), st.h(), -1, 2);
    }
    flow = m_noise;
    flow = gpu_stgauss2_filter(flow, st, 6, 22.5f, false, true, true, 2, 1.0f, 2);
    flow = gpu_stgauss2_filter(flow, st, 1, 22.5, false, true, true, 2, 1.0f, 2 );

    gpu_image<float4> img = lab;
    gpu_image<float4> Ie = img;
    gpu_image<float4> Ia = img;

    if (enable_bf) {
        int N = std::max(n_e, n_a);
        for (int i = 0; i < N; ++i) {
            if (filter_type == "oa") {
                img = gpu_oabf_1d(img, lfm, sigma_dg, sigma_rg, false, precision_g);
                img = gpu_oabf_1d(img, lfm, sigma_dt, sigma_rt, true, precision_t);
            } else if (filter_type == "xy") {
                img = gpu_bilateral_filter_xy(img, sigma_dg, sigma_rg);
            } else if (filter_type == "fbl") {
                img = gpu_oabf_1d(img, lfm, sigma_dg, sigma_rg, false, precision_g);
                img = gpu_stbf2_filter(img, st, sigma_dt, sigma_rt, precision_t, 90.0f, false, true, true, 2, 1);
            } else {
                img = gpu_bilateral_filter(img, sigma_dg, sigma_rg);
            }
            if (i == (n_e - 1)) Ie = img;
            if (i == (n_a - 1)) Ia = img;
        }
    }

    gpu_image<float> L;
    if (output != "fill") {
        L = gpu_shuffle(Ie, 0);
        if (dog_type == "flow-based") {
            if (!dog_reparam)
                L = gpu_gradient_dog( L, lfm, sigma_e, dog_k, dog_tau, precision_e );
            else
                L = gpu_gradient_dog2( L, lfm, sigma_e, dog_k, dog_p, precision_e );

            if (dog_fgauss == "euler") {
                L = gpu_stgauss2_filter(L, st, sigma_m, 90, false, false, false, 1, step_m, precision_m);
            } else if (dog_fgauss == "rk2-nn") {
                L = gpu_stgauss2_filter(L, st, sigma_m, 90, false, false, false, 2, step_m, precision_m);
            } else if (dog_fgauss == "rk2") {
                L = gpu_stgauss2_filter(L, st, sigma_m, 90, false, true, true, 2, step_m, precision_m);
            } else if (dog_fgauss == "rk4") {
                L = gpu_stgauss2_filter(L, st, sigma_m, 90, false, true, true, 4, step_m, precision_m);
            }
        } else {
            if (!dog_reparam)
                L = gpu_isotropic_dog( L, sigma_e, dog_k, dog_tau, precision_e );
            else
                L = gpu_isotropic_dog2( L, sigma_e, dog_k, dog_p, precision_e );

        }

        {
            float eps, phi;
            if (!dog_reparam) {
                eps = dog_eps;
                phi = dog_phi;
            } else {
                eps = dog_eps_p;

#ifndef ANGLE_PHI
                phi = dog_phi_p;
#else
                phi = .tan( M_PI*(dog_phi_p/180) );
                //phi = .01*tan( M_PI*(dog_phi_p/180) );
#endif
            }

            if (dog_adj_func == "smoothstep")  {
                L = gpu_dog_threshold_smoothstep(L, eps, phi);
            } else {
                L = gpu_dog_threshold_tanh(L, eps, phi);
            }
        }
    }

    if (quantization) {
        if (quant_type == "fixed") {
            Ia = gpu_wog_luminance_quant( Ia, nbins, phi_q );
        } else {
            Ia = gpu_wog_luminance_quant( Ia, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi );
        }
    }

    img = gpu_lab2rgb(Ia);

    if (output == "edges") {
        img = gpu_l2rgb(gpu_mul(L,100));
    } else if (output == "fill+edges") {
        img = gpu_blend_intensity(img, L, GPU_BLEND_MULTIPLY);
    }
    if (input_gamma == "linear-rgb") {
        img = gpu_srgb2linear(img);
    }

    if (warp_sharp) {
        img = gpu_wog_warp_sharp(img, sigma_w, precision_w, phi_w);
    }

    if (final_smooth) {
       if (final_type == "3x3")
           img = gpu_gauss_filter_3x3(img);
       else if (final_type == "5x5")
           img = gpu_gauss_filter_5x5(img);
       if (final_type == "flow-nearest") {
            img = gpu_stgauss2_filter( img, st, sigma_a, 90, false, false, false, 2, 1.0f, 2);
       } else {
            img = gpu_stgauss2_filter( img, st, sigma_a, 90, false, true, true, 2, 1.0f, 2);
       }
    }

    m_result[0] = gpu_image_to_qimage(img);
    m_result[1] = qsrc;
    m_result[2] = gpu_image_to_qimage(flow);

    imageChanged(image());

    qDebug() << AbstractParam::paramText(this);
}


void MainWindow::dogChanged() {
    dog_tau_g->setValue(!dog_reparam);
    dog_p_g->setValue(dog_reparam);
    if (!dog_reparam) {
        dog_p_ptr->setValue( dog_tau / (1 - dog_tau) );
        dog_eps_p_ptr->setValue( dog_eps / (1 - dog_tau) );

#ifndef ANGLE_PHI
        dog_phi_p_ptr->setValue( dog_phi * (1 - dog_tau) );
#else
        dog_phi_p_ptr->setValue( atan(dog_phi * (1 - dog_tau)) / M_PI*180 );
        //dog_phi_p_ptr->setValue( atan( 100 * dog_phi * (1 - dog_tau)) / M_PI*180 );
#endif

    } else {
        dog_tau_ptr->setValue( dog_p / (1 + dog_p) );
        dog_eps_ptr->setValue( dog_eps_p / (1 + dog_p) );
        dog_phi_ptr->setValue( dog_phi_p * (1 + dog_p) );
    }
}


void MainWindow::onIndexChanged(int index) {
    imageChanged(image());
}


void MainWindow::onVideoChanged(int nframes) {
    gpu_cache_clear();
    window()->setWindowFilePath(m_player->filename());
    window()->setWindowTitle(m_player->filename() + "[*] - " + qApp->applicationName());
    actionRecord->setEnabled(nframes > 1);
}


void MainWindow::on_actionSavePNG_triggered() {
    m_imageView->savePNG(AbstractParam::paramText(this));
}


void MainWindow::on_actionLoadSettings_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + ".ini");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getOpenFileName(this, "Load Settings", filename,
        "INI Format (*.ini);;All files (*.*)");
    if (!filename.isEmpty()) {
        QSettings iniFile(filename, QSettings::IniFormat);
        AbstractParam::restoreSettings(iniFile, this);
        settings.setValue("savename", filename);
    }
}


void MainWindow::on_actionSaveSettings_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + ".ini");
        filename  = fn.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getSaveFileName(this, "Save Settings", filename,
        "INI Format (*.ini);;All files (*.*)");
    if (!filename.isEmpty()) {
        QSettings iniFile(filename, QSettings::IniFormat);
        iniFile.clear();
        AbstractParam::saveSettings(iniFile, this);
        settings.setValue("savename", filename);
    }
}


void MainWindow::on_actionShowSettings_triggered() {
    QSettings settings;
    QString inputPath = window()->windowFilePath();
    QString outputPath = settings.value("savename", inputPath).toString();

    QString filename;
    QFileInfo fi(inputPath);
    QFileInfo fo(outputPath);
    if (!fi.baseName().isEmpty()) {
        QFileInfo fn(fo.dir(), fi.baseName() + "-out.png");
        filename  = fi.absoluteFilePath();
    } else {
        filename  = fo.absolutePath();
    }

    filename = QFileDialog::getOpenFileName(this, "Open PNG", filename,
        "PNG Format (*.png);;All files (*.*)");
    if (!filename.isEmpty()) {
        QImage image(filename);
        if (image.isNull()) {
            QMessageBox::critical(this, "Error", QString("Info PNG '%1' failed!").arg(filename));
            return;
        }
        QString text = image.text("Description");
        text.replace(";", ";\n");
        QMessageBox::information(this, "Show Info", text);
    }
}


