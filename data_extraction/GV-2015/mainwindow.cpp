/***************************************************************************
**                                                                        **
**  QCustomPlot, an easy to use, modern plotting widget for Qt            **
**  Copyright (C) 2011-2015 Emanuel Eichhammer                            **
**                                                                        **
**  This program is free software: you can redistribute it and/or modify  **
**  it under the terms of the GNU General Public License as published by  **
**  the Free Software Foundation, either version 3 of the License, or     **
**  (at your option) any later version.                                   **
**                                                                        **
**  This program is distributed in the hope that it will be useful,       **
**  but WITHOUT ANY WARRANTY; without even the implied warranty of        **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         **
**  GNU General Public License for more details.                          **
**                                                                        **
**  You should have received a copy of the GNU General Public License     **
**  along with this program.  If not, see http://www.gnu.org/licenses/.   **
**                                                                        **
****************************************************************************
**           Author: Emanuel Eichhammer                                   **
**  Website/Contact: http://www.qcustomplot.com/                          **
**             Date: 25.04.15                                             **
**          Version: 1.3.1                                                **
****************************************************************************/

/************************************************************************************************************
**                                                                                                         **
**  This is the example code for QCustomPlot.                                                              **
**                                                                                                         **
**  It demonstrates basic and some advanced capabilities of the widget. The interesting code is inside     **
**  the "setup(...)Demo" functions of MainWindow.                                                          **
**                                                                                                         **
**  In order to see a demo in action, call the respective "setup(...)Demo" function inside the             **
**  MainWindow constructor. Alternatively you may call setupDemo(i) where i is the index of the demo       **
**  you want (for those, see MainWindow constructor comments). All other functions here are merely a       **
**  way to easily create screenshots of all demos for the website. I.e. a timer is set to successively     **
**  setup all the demos and make a screenshot of the window area and save it in the ./screenshots          **
**  directory.                                                                                             **
**                                                                                                         **
*************************************************************************************************************/

#include "include/mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QDebug>
#include <QDesktopWidget>
#include <QScreen>
#include <QMessageBox>
#include <QMetaEnum>


MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  speed = 3.0;
  elevation = 0.0;
  muscle = 1;
  start = 0;
  ms_count = 0;
  gait_cycle_p = 0;

  ui->setupUi(this);
  setGeometry(400, 250, 542, 390);

  setupDemo(10);

  // for making screenshots of the current demo or all demos (for website screenshots):
  //QTimer::singleShot(1500, this, SLOT(allScreenShots()));
  //QTimer::singleShot(4000, this, SLOT(screenShot()));

}


void MainWindow::setupDemo(int demoIndex)
{
  switch (demoIndex)
  {

    case 10: setupRealtimeDataDemo(ui->customPlot, ui->customPlot_2, ui->customPlot_3, ui->customPlot_4); break;

  }

  /* Init MEP */
  MainWindow::MEP_model.calcWeightings(elevation,speed);
  MainWindow::MEP_model.calcMEP(elevation,speed);
  
  MainWindow::MEP_model.saveXP();
  MainWindow::MEP_model.saveMEP();
  MainWindow::MEP_model.saveWeightings();

  setWindowTitle("MEP model demo");
  statusBar()->clearMessage();
  currentDemoIndex = demoIndex;
  ui->customPlot->replot();
  ui->customPlot_2->replot();
  ui->customPlot_3->replot();
  ui->customPlot_4->replot();
  ui->counter_label->setText("0");
  
  
}

void MainWindow::setupRealtimeDataDemo(QCustomPlot *customPlot, 
	QCustomPlot *customPlot_2, QCustomPlot *customPlot_3, 
	QCustomPlot *customPlot_4)
{
#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
  QMessageBox::critical(this, "", "You're using Qt < 4.7, the realtime data demo needs functions that are available with Qt 4.7 to work properly");
#endif
  demoName = "Real Time Data Demo";
  
  // include this section to fully disable antialiasing for higher performance:
  /*
  customPlot->setNotAntialiasedElements(QCP::aeAll);
  QFont font;
  font.setStyleStrategy(QFont::NoAntialias);
  customPlot->xAxis->setTickLabelFont(font);
  customPlot->yAxis->setTickLabelFont(font);
  customPlot->legend->setFont(font);
  */
  customPlot->addGraph(); // blue line
  customPlot->graph(0)->setPen(QPen(Qt::blue));
  customPlot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot->graph(0)->setAntialiasedFill(false);
  
  customPlot->addGraph(); // blue dot
  customPlot->graph(1)->setPen(QPen(Qt::blue));
  customPlot->graph(1)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(1)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot->addGraph(); // red dot
  customPlot->graph(2)->setPen(QPen(Qt::red));
  customPlot->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot->xAxis->setAutoTickStep(false);
  customPlot->xAxis->setTickStep(2);
  customPlot->axisRect()->setupFullAxesBox();
  
  // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->yAxis2, SLOT(setRange(QCPRange)));
  
  //------
  
  customPlot_2->addGraph(); // blue line
  customPlot_2->graph(0)->setPen(QPen(Qt::blue));
  customPlot_2->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot_2->graph(0)->setAntialiasedFill(false);
  
  customPlot_2->addGraph(); // blue dot
  customPlot_2->graph(1)->setPen(QPen(Qt::blue));
  customPlot_2->graph(1)->setLineStyle(QCPGraph::lsNone);
  customPlot_2->graph(1)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_2->addGraph(); // red dot
  customPlot_2->graph(2)->setPen(QPen(Qt::red));
  customPlot_2->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot_2->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_2->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot_2->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot_2->xAxis->setAutoTickStep(false);
  customPlot_2->xAxis->setTickStep(2);
  customPlot_2->axisRect()->setupFullAxesBox();
  
   // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot_2->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot_2->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->yAxis2, SLOT(setRange(QCPRange)));
  
  //------
  
  customPlot_3->addGraph(); // blue line
  customPlot_3->graph(0)->setPen(QPen(Qt::blue));
  customPlot_3->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot_3->graph(0)->setAntialiasedFill(false);
  
  customPlot_3->addGraph(); // blue dot
  customPlot_3->graph(1)->setPen(QPen(Qt::blue));
  customPlot_3->graph(1)->setLineStyle(QCPGraph::lsNone);
  customPlot_3->graph(1)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_3->addGraph(); // red dot
  customPlot_3->graph(2)->setPen(QPen(Qt::red));
  customPlot_3->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot_3->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_3->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot_3->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot_3->xAxis->setAutoTickStep(false);
  customPlot_3->xAxis->setTickStep(2);
  customPlot_3->axisRect()->setupFullAxesBox();
  
   // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot_3->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot_3->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->yAxis2, SLOT(setRange(QCPRange)));
  
  //------
  customPlot_4->addGraph(); // blue line
  customPlot_4->graph(0)->setPen(QPen(Qt::blue));
  customPlot_4->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
  customPlot_4->graph(0)->setAntialiasedFill(false);
  
  customPlot_4->addGraph(); // blue dot
  customPlot_4->graph(1)->setPen(QPen(Qt::blue));
  customPlot_4->graph(1)->setLineStyle(QCPGraph::lsNone);
  customPlot_4->graph(1)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_4->addGraph(); // red dot
  customPlot_4->graph(2)->setPen(QPen(Qt::red));
  customPlot_4->graph(2)->setLineStyle(QCPGraph::lsNone);
  customPlot_4->graph(2)->setScatterStyle(QCPScatterStyle::ssDisc);
  
  customPlot_4->xAxis->setTickLabelType(QCPAxis::ltDateTime);
  customPlot_4->xAxis->setDateTimeFormat("hh:mm:ss");
  customPlot_4->xAxis->setAutoTickStep(false);
  customPlot_4->xAxis->setTickStep(2);
  customPlot_4->axisRect()->setupFullAxesBox();
  
  // make left and bottom axes transfer their ranges to right and top axes:
  connect(customPlot_4->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->xAxis2, SLOT(setRange(QCPRange)));
  connect(customPlot_4->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot_2->yAxis2, SLOT(setRange(QCPRange)));
  
  
  // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
  connect(&dataTimer, SIGNAL(timeout()), this, SLOT(realtimeDataSlot()));
  dataTimer.start(1); // Interval 0 means to refresh as fast as possible
}


void MainWindow::realtimeDataSlot()
{

  // calculate two new data points:
//#if QT_VERSION < QT_VERSION_CHECK(4, 7, 0)
//  double key = 0;
//#else
  
  //std::cout << key << endl;
//#endif
  if (start == 1)
  {
	  
  ms_count++; 
 
  double key = (double)(ms_count)/1000.0;//QDateTime::currentDateTime().toMSecsSinceEpoch()/10000.0;
  	  
  static double lastPointKey = 0;
  double set_cycle_speed = (3600/(speed * 1000))/200;
  if (key-lastPointKey > set_cycle_speed) // at most add point every 10 ms
  {

        double data;

        double value0 = MEP_model.getMEP(gait_cycle_p,"VastMed");
        double value1 = MEP_model.getMEP(gait_cycle_p,"Sol");
        double value2 = MEP_model.getMEP(gait_cycle_p,"TA");
        double value3 = MEP_model.getMEP(gait_cycle_p,"Semi");
        
        ui->counter_label-> text() = gait_cycle_p;
        QString gc = QString::number(gait_cycle_p);
        ui->counter_label->setText(gc);

        // add data to lines:
        ui->customPlot->graph(0)->addData(key, value0);
        // set data of dots:
        ui->customPlot->graph(1)->clearData();
        ui->customPlot->graph(1)->addData(key, value0);
        if (gait_cycle_p == 0)
			ui->customPlot->graph(2)->addData(key, value0);
        // remove data of lines that's outside visible range:
        ui->customPlot->graph(0)->removeDataBefore(key-8);
        // rescale value (vertical) axis to fit the current data:
        ui->customPlot->graph(0)->rescaleValueAxis();
        lastPointKey = key;

        // make key axis range scroll with the data (at a constant range size of 8):
        ui->customPlot->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
        ui->customPlot->replot();
        
        // add data to lines:
        ui->customPlot_2->graph(0)->addData(key, value1);
        // set data of dots:
        ui->customPlot_2->graph(1)->clearData();
        ui->customPlot_2->graph(1)->addData(key, value1);
        if (gait_cycle_p == 0)
			ui->customPlot_2->graph(2)->addData(key, value1);
        // remove data of lines that's outside visible range:
        ui->customPlot_2->graph(0)->removeDataBefore(key-8);
        // rescale value (vertical) axis to fit the current data:
        ui->customPlot_2->graph(0)->rescaleValueAxis();
        
        // make key axis range scroll with the data (at a constant range size of 8):
        ui->customPlot_2->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
        ui->customPlot_2->replot();
        
        // add data to lines:
        ui->customPlot_3->graph(0)->addData(key, value2);
        // set data of dots:
        ui->customPlot_3->graph(1)->clearData();
        ui->customPlot_3->graph(1)->addData(key, value2);
        if (gait_cycle_p == 0)
			ui->customPlot_3->graph(2)->addData(key, value2);
        // remove data of lines that's outside visible range:
        ui->customPlot_3->graph(0)->removeDataBefore(key-8);
        // rescale value (vertical) axis to fit the current data:
        ui->customPlot_3->graph(0)->rescaleValueAxis();
        
        // make key axis range scroll with the data (at a constant range size of 8):
        ui->customPlot_3->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
        ui->customPlot_3->replot();
        
        // add data to lines:
        ui->customPlot_4->graph(0)->addData(key, value3);
        // set data of dots:
        ui->customPlot_4->graph(1)->clearData();
        ui->customPlot_4->graph(1)->addData(key, value3);
        if (gait_cycle_p == 0)
			ui->customPlot_4->graph(2)->addData(key, value3);
        // remove data of lines that's outside visible range:
        ui->customPlot_4->graph(0)->removeDataBefore(key-8);
        // rescale value (vertical) axis to fit the current data:
        ui->customPlot_4->graph(0)->rescaleValueAxis();
        
        // make key axis range scroll with the data (at a constant range size of 8):
        ui->customPlot_4->xAxis->setRange(key+0.25, 8, Qt::AlignRight);
        ui->customPlot_4->replot();
        
        if (gait_cycle_p > 198)
            gait_cycle_p = 0;
        else
            gait_cycle_p++;
        
      }
              
    // calculate frames per second:
    static double lastFpsKey;
    static int frameCount;
    ++frameCount;
    if (key-lastFpsKey > 2) // average fps over 2 seconds
    {
        ui->statusBar->showMessage(
            QString("%1 FPS, Total Data points: %2")
            .arg(frameCount/(key-lastFpsKey), 0, 'f', 0)
            .arg(ui->customPlot->graph(0)->data()->count()+ui->customPlot->graph(1)->data()->count())
            , 0);
        lastFpsKey = key;
        frameCount = 0;
    }
  }
}


void MainWindow::setupPlayground(QCustomPlot *customPlot)
{
  Q_UNUSED(customPlot)
}

MainWindow::~MainWindow()
{
  delete ui;
}


void MainWindow::on_startButton_released()
{
    MainWindow::MEP_model.calcWeightings(elevation,speed);
    MainWindow::MEP_model.calcMEP(elevation,speed);
    start = 1;
}

void MainWindow::on_stopButton_released()
{
    start = 0;
}

void MainWindow::on_comboBox_currentIndexChanged(int index)
{

}

void MainWindow::on_speedEdit_returnPressed()
{

    speed = ui->speedEdit->text().toDouble();
    MainWindow::MEP_model.calcWeightings(elevation,speed);
    MainWindow::MEP_model.calcMEP(elevation,speed);
}

void MainWindow::on_elevationEdit_returnPressed()
{
    elevation = ui->elevationEdit->text().toDouble();
    MainWindow::MEP_model.calcWeightings(elevation,speed);
    MainWindow::MEP_model.calcMEP(elevation,speed);
}

void MainWindow::on_heel_strike_p_released()
{
    gait_cycle_p = 0;
}

void MainWindow::on_toe_off_p_released()
{
    gait_cycle_p = 120;
}
