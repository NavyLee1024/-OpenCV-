## 图像的掩膜操作：

- 判断一个图像有没有存入一个Mat src容器中，可以通过判断src.empty或者src.data属性来看是。

```c++
if(!src.data) {  //或者if (src.empty())
    printf("could not load image...\n");  
    return  -1;  
}  
```

- Mat对象中数据的行指针获取，并且根据这个行指针可以逐行取每个像素。其实本质就是把图像看成一个二维数组，每个二维数组的行数组（一维数组）就是该行数据的数组名或者就是行指针。

```c++
const uchar* previous = src.ptr<uchar>(8);  
```

- 像素范围处理函数，如果这个像素值是float，或者int型，就是强行限制一个像素的值（某个通道字节）到0~255之间。小于0的归为0，大于255的统一到255，之间的数还是其本身。而最后都要转换成BGR888显示，就需要对像素值进行压缩。下面假设是单通道的。

```c++
uchar output = saturate_cast<uchar>(1025);//output的值最终会为255  
```

- 掩膜操作的本质就是定义一个小的图像窗口去卷积整幅图像。本质上就是一种滤波操作。这个小窗口就是卷积核。上述的过程也可以自己写。下面演示用filter2D函数来操作。其实后面会说到各种滤波卷积操作，都可以用filter2D函数来实现。可以根据实际需求来选择不通用法。


```c++
Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);//定义一个卷积核，本质也是一个图像  
filter2D(src, dst, src.depth(), kernel);//卷积原始图像后输出结果图像dst，src.depth()表述位图的深度，8,16,24,32等等
namedWindow("contrast image demo", CV_WINDOW_AUTOSIZE);  
imshow("contrast image demo", dst);  
```

## Mat对象

- Mat对象的构造函数和常用方法有如下一下，mat的构造方法有多种方式，后期用到了可以再细看。这里会涉及到一个叫Scalar（B,G,R,A）的标量，一般用它定义色彩。可以是BGRA,也可以BGR，也可以其他形式的色彩方式。这个主要跟通道数有关。如果初始化一个有颜色的Mat对象，就可以用如下带scalar形参的构造函数来实现。或者以后用到的画线，画框等操作，都需要先定义画笔的颜色。：


```c++
//假设有两个Mat src  ,  dst;对象  构造方式有很多，这里不写了。如下图
Mat src  ,  dst;//他两的构造函数也有多个形式  
 void  copyTo(Mat mat) //用法：  src.copyTo(dts)  
 void  convertTo(Mat dst,  int  type) //用法：src.convertTo(dst,CV_BGR2GRAY)    
Mat clone() //用法：  dst = src.clone()  
 int  channels() //用法：  src.channels()  
 int  depth() //用法：  int depth = src.depth()  
 bool  empty(); //用法：  bool a = src.empty()  
uchar* ptr(i=0) //用法：  uchar *rowAddress = src.ptr  
```

- Mat对象的拷贝，赋值使用有讲究。如果只是用“=”或者构造函数传递的话，一般只是部分复制mat的头和指针部分。不会复制数据。只有通过clone，copyTo等操作才能算复制成完整的两个不同对象。


```c++
Mat B(A)  // 只复制头部和指针给B，不复制数据部分  
Mat F = A.clone(); 或 Mat G; A.copyTo(G);//完全复制 
```

- 创建多维数组先不做笔记，用的少！

- 定义小数组一般用于定义 卷积核等等

```c++
Mat C = (Mat_< double >(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);   //double可以改成uchar int等等  
cout << "C = " << endl << " " << C << endl << endl;  
```

## 图像操作

图像的操作，尽量不要涉及到内存的操作，甚至物理内存的操作。这些都是高手老司机的技能。否则操作不当容易导致各种系统报错。下面主要说一些基于API的像素读写修改操作。

###  读写像素 

 读一个GRAY像素点的像素值（CV_8UC1）每个像素值用Scalar类型存放。

```c++
Scalar intensity = img.at<uchar>(y, x); 
```

或者 

```c++
Scalar intensity = img.at<uchar>(Point(x, y));
```

读一个RGB像素点的像素值

```c++
Vec3f intensity = img.at<Vec3f>(y, x); 
float blue = intensity.val[0]; 
float green = intensity.val[1]; 
float red = intensity.val[2];
```

###  修改像素值 

灰度图像上，指定某个点坐标修改其值为128

```c++
img.at<uchar>(y, x) = 128;
//RGB三通道图像指定某个点坐标修改其值为128
img.at<Vec3b>(y,x)[0]=128; // blue 有顺序
img.at<Vec3b>(y,x)[1]=128; // green
img.at<Vec3b>(y,x)[2]=128; // red
```

```c++
//空白图像整体赋值
img = Scalar(0);
ROI选择（感兴趣区域）
Rect r(10, 10, 100, 100); 
Mat smallImg = img(r);
```

 Vec3b与Vec3F 

Vec3b对应三通道的顺序是blue、green、red的uchar类型数据。

Vec3f对应三通道的float类型数据

把CV_8UC1转换到CV32F1实现如下：

```c++
src.convertTo(dst, CV_32F);
```

###  补充： 反色定义

就是用255去减去图片上每个像素点的值，会出现如同交卷底片的感觉。相当于取反操作bitwise_not（）；

###  补充： 取灰度的方法

也有很多，可以同max,min取RGB中最大最小值，再付给每个像素的三个通道也是可以的。或者各种库函数操作，后面会有说到。

###  补充： 图像的基本运算

有很多种，比如两幅图像可以相加、相减、相乘、相除、位运算、平方根、对数、绝对值等；图像也可以放大、缩小、旋转，还可以截取其中的一部分作为ROI（感兴趣区域）进行操作，各个颜色通道还可以分别提取及对各个颜色通道进行各种运算操作。总之，对于图像可以进行的基本运算非常的多，只是挑了些常用的操作详解。

```c++
void  add(InputArray src1, InputArray src2, OutputArray dst,InputArray mask=noArray(),  int  dtype=-1);//dst = src1 + src2  

 void  subtract(InputArray src1, InputArray src2, OutputArray dst,InputArray mask=noArray(),  int  dtype=-1);//dst = src1 - src2  

void multiply(InputArray src1, InputArray src2,OutputArray dst,  double  scale=1,  int  dtype=-1);//dst = scale*src1*src2  

 void  divide(InputArray src1, InputArray src2, OutputArray dst, double  scale=1,  int  dtype=-1);//dst = scale*src1/src2  

 void  divide( double  scale, InputArray src2,OutputArray dst,  int  dtype=-1);//dst = scale/src2  

 void  scaleAdd(InputArray src1,  double  alpha, InputArray src2, OutputArray dst);//dst = alpha*src1 + src2  

 void  addWeighted(InputArray src1,  double  alpha, InputArray src2, double  beta,  double  gamma, OutputArray dst,  int  dtype=-1);//dst = alpha*src1 + beta*src2 + gamma  

 void  sqrt(InputArray src, OutputArray dst);//计算每个矩阵元素的平方根  

 void  pow(InputArray src,  double  power, OutputArray dst);//src的power次幂  

  void  exp(InputArray src, OutputArray dst);//dst = e src（ 表示指数的意思）  

void  log(InputArray src, OutputArray dst);//dst = log(abs(src))  

//上述的基本操作中都属于将基础数学运算应用于图像像素的处理中，下面将着重介绍

bitwise_and、bitwise_or、bitwise_xor、bitwise_not这四个按位操作函数。  

 void  bitwise_and(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());//dst = src1 & src2  （灰度图像或彩色图像均可）

 void  bitwise_or(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());//dst = src1 | src2  （灰度图像或彩色图像均可）

 void  bitwise_xor(InputArray src1, InputArray src2,OutputArray dst, InputArray mask=noArray());//dst = src1 ^ src2  （灰度图像或彩色图像均可）

 void  bitwise_not(InputArray src, OutputArray dst,InputArray mask=noArray());//dst = ~src
```

###  补充： 图像掩膜（mask）

- 在有些图像处理的函数中有的参数里面会有mask参数，即此函数支持掩膜操作，首先何为掩膜以及有什么用，如下：数字图像处理中的掩膜的概念是借鉴于PCB制版的过程，在半导体制造中，许多芯片工艺步骤采用光刻技术，用于这些步骤的图形“底片”称为掩膜（也称作“掩模”），其作用是：在硅片上选定的区域中对一个不透明的图形模板遮盖，继而下面的腐蚀或扩散将只影响选定的区域以外的区域。

- 图像掩膜与其类似，用选定的图像、图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。


- 数字图像处理中,掩模为二维矩阵数组,有时也用多值图像，图像掩模主要用于：

①提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。

②屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。

③结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。

④特殊形状图像的制作。

在所有图像基本运算的操作函数中，凡是带有掩膜（mask）的处理函数，其掩膜都参与运算（输入图像运算完之后再与掩膜图像或矩阵运算）。

- 掩膜应用实例


```c++
//下为脸部面具图，背景为白色，利用按位操作及掩膜技术清晰抠出面具轮廓。
// 转换面具为灰度图像  
cvtColor(faceMaskSmall, grayMaskSmall, CV_BGR2GRAY);  

// 隔离图像上像素的边缘，仅与面具有关（即面具的白色区域剔除），下面函数将大于230像素的值置为0,小于的置为255 
threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, CV_THRESH_BINARY_INV);  

// 通过反转上面的图像创建掩码（因为不希望背景影响叠加）  
bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);  

//使用位“与”运算来提取面具精确的边界  
bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh); 

// 使用位“与”运算来叠加面具  
bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv); 
```

## 图像混合

### 理论-线性混合操作 :

​	g(x)=(1-a)*f0(x) + a*f1(x)

​	其中 a 的取值范围为0~1之间 ,x如果放在二维图像数值中应该当做（x,y）看，就是每个点的坐标。f0，f1相当于两幅图像。g(x)就是新的图像。个人感觉有点像互补滤波。

​	如果每个像素点的值都在255之间，那么这样的相加（混合，融合）后的结果值也不会超过255.所以在OpenCV中的名称叫addWeighted，类似两幅图有不同的加权重系数。

​	opencv中还有个add函数，这个就是强行相加。不会有两幅图叠加显示的虚幻效果的。

```c
//参数1：输入图像Mat – src1
//参数2：输入图像src1的alpha值
//参数3：输入图像Mat – src2
//参数4：输入图像src2的alpha值
//参数5：gamma值
//参数6：输出混合图像
//注意点：两张图像的大小和类型必须一致才可以
addWeighted(src1, alpha, src2, (1.0 - alpha), 0.0, dst);//0.0是gamma调整参数
```

## 调整图像的亮度与对比度

```c++
lMat new_image = Mat::zeros( image.size(), image.type() );  //创建一张跟原图像大小和类型一致的空白图像、像素值初始化为0
l saturate_cast<uchar>(value)//确保值大小范围为0~255之间
lMat.at<Vec3b>(y,x)[index]=value //给每个像素点每个通道赋值
```

```c++
#include <iostream>
/*简单的像素亮度和对比度调节*/
using namespace cv;
int main(int argc, char** argv) {
	Mat src, dst;
	src = imread("D:/vcprojects/images/test.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	char input_win[] = "input image";
	cvtColor(src, src, CV_BGR2GRAY);//变成灰度图像
	namedWindow(input_win, CV_WINDOW_AUTOSIZE);
	imshow(input_win, src);

	// contrast and brigthtness changes 
	int height = src.rows;
	int width = src.cols;
	dst = Mat::zeros(src.size(), src.type());
	float alpha = 1.2;
	float beta = 30;

	Mat m1;
	src.convertTo(m1, CV_32F);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (src.channels() == 3) {
				float b = m1.at<Vec3f>(row, col)[0];// blue
				float g = m1.at<Vec3f>(row, col)[1]; // green
				float r = m1.at<Vec3f>(row, col)[2]; // red

				// output  亮度
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
			}
			else if (src.channels() == 1) {
				float v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = saturate_cast<uchar>(v*alpha + beta);
			}
		}
	}
```

## 绘制图形与文字

### 使用cv::Point与cv::Scalar

```c++
//Point表示2D平面上一个点x,y
  Point p;
  p.x = 10;
  p.y = 8;
   //or
  p = Pont(10,8);
//Scalar表示四个元素的向量
  Scalar(a, b, c);// a = blue, b = green, c = red表示RGB三个通道
```

### 绘制线、矩形、园、椭圆等基本几何形状

```c
//画线 cv::line （LINE_4\LINE_8\LINE_AA）//LINE_AA表示反锯齿，其他两个不能反锯齿。
//画椭圆cv::ellipse
//画矩形cv::rectangle
//画圆cv::circle
//画填充cv::fillPoly

```

```c++
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
Mat bgImage;
const char* drawdemo_win = "draw shapes and text demo";
void MyLines();
void MyRectangle();
void MyEllipse();
void MyCircle();
void MyPolygon();
void RandomLineDemo();
int main(int argc, char** argv) {
	bgImage = imread("D:/vcprojects/images/test1.png");
	if (!bgImage.data) {
		printf("could not load image...\n");
		return -1;
	}
	//MyLines();
	//MyRectangle();
	//MyEllipse();
	//MyCircle();
	//MyPolygon();

    //在图像上显示文字， CV_FONT_HERSHEY_COMPLEX==字体  1.0表示大小  3表示粗细
	//putText(bgImage, "Hello OpenCV", Point(300, 300), CV_FONT_HERSHEY_COMPLEX, 1.0, Scalar(12, 23, 200), 3, 8);
	//namedWindow(drawdemo_win, CV_WINDOW_AUTOSIZE);
	//imshow(drawdemo_win, bgImage);

	RandomLineDemo();
	waitKey(0);
	return 0;
}

void MyLines() {
	Point p1 = Point(20, 30);
	Point p2;
	p2.x = 400;
	p2.y = 400;
	Scalar color = Scalar(0, 0, 255);
	line(bgImage, p1, p2, color, 1, LINE_AA);//两点一线 反锯齿
}

void MyRectangle() {
	Rect rect = Rect(200, 100, 300, 300);//坐标点在（200,100）处，长宽值为300,300的正方形
	Scalar color = Scalar(255, 0, 0);
	rectangle(bgImage, rect, color, 2, LINE_8);//两个对角点能确定一个矩形
}

void MyEllipse() {//椭圆
	Scalar color = Scalar(0, 255, 0);
	ellipse(bgImage, Point(bgImage.cols / 2, bgImage.rows / 2), Size(bgImage.cols / 4, bgImage.rows / 8), 90, 0, 360, color, 2, LINE_8);
}

void MyCircle() {//圆形
	Scalar color = Scalar(0, 255, 255);
	Point center = Point(bgImage.cols / 2, bgImage.rows / 2);
	circle(bgImage, center, 150, color, 2, 8); //背景图  圆心 半径  线宽  类型？？？（锯齿）
}

void MyPolygon() {//多边形 填充图
	Point pts[1][5];//二维数组
	pts[0][0] = Point(100, 100);
	pts[0][1] = Point(100, 200);
	pts[0][2] = Point(200, 200);
	pts[0][3] = Point(200, 100);
	pts[0][4] = Point(100, 100);

	const Point* ppts[] = { pts[0] };//二维数组的行数组名
	int npt[] = { 5 };
	Scalar color = Scalar(255, 12, 255);

	fillPoly(bgImage, ppts, npt, 1, color, 8);
}

void RandomLineDemo() {//随机函数
	RNG rng(12345);//opencv中的随机函数 给一个足够大的“种子” 伪随机
	Point pt1;
	Point pt2;
	Mat bg = Mat::zeros(bgImage.size(), bgImage.type());
	namedWindow("random line demo", CV_WINDOW_AUTOSIZE);
	for (int i = 0; i < 100000; i++) {//循环一段时间
		pt1.x = rng.uniform(0, bgImage.cols);//x值在0~bgImage.cols之间
		pt2.x = rng.uniform(0, bgImage.cols);
		pt1.y = rng.uniform(0, bgImage.rows);
		pt2.y = rng.uniform(0, bgImage.rows);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));//色彩随机
		if (waitKey(50) > 0) {
			break;
		}
		line(bg, pt1, pt2, color, 1, 8);
		imshow("random line demo", bg);
	}
}

```
## 模糊图像（滤波）

### 	滤波方法类型：

​	模糊(blur)  平滑(smooth)   滤波    掩膜   等等都差不多。本质就是用一个小算子去卷积整个图片。可以起到去噪效果。常用的滤波方法有如下几个：

​	均值滤波:无法克服边缘像素信息丢失的缺陷。原因是基于平均权重。 

​		blur(Mat src, Mat dst, Size(xradius, yradius), Point(-1,-1)); :

​	高斯滤波： 基于正态分布的算子，基于权重的模糊。部分做到了边缘像素不模糊，但是不完全。后面会对该方式完善——高斯双边模糊。

​		GaussianBlur(Mat src, Mat dst, Size(11, 11), sigmax, sigmay); 

​	中值滤波：对卷积核窗口中的数据进行冒泡排序，后取中间值替换核中间的值。对椒盐噪声有很好的抑制效果。衍生出来可以取最小值/最大值滤波。后面会用到形态学的膨胀和腐蚀中。

​		lmedianBlur（Mat src, Mat dest, ksize） 

​	高斯双边模糊：l是边缘保留的滤波方法，避免了边缘信息丢失，保留了图像轮廓不变 （常用于美颜的磨皮滤镜，因为保留了边缘信息同时平滑肤色）

​		lbilateralFilter(src, dest, d=15, 150, 3); i

### 代码演示：

```c++
#include <opencv2/opencv.hpp> 
#include <iostream> 
using namespace cv;

int main(int argc, char** argv) {
	Mat src, dst;
	src = imread("D:/vcprojects/images/test.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	char input_title[] = "input image";
	char output_title[] = "blur image";
	namedWindow(input_title, CV_WINDOW_AUTOSIZE);
	namedWindow(output_title, CV_WINDOW_AUTOSIZE);
	imshow(input_title, src);
	
    //Size(15, 1)==水平方向模糊  Size(1, 15)垂直方向模糊，注意算子都要是奇数值。
    //也可以用filter2D来做模糊，这个最通用。但是要自定义一个算子。
	blur(src, dst, Size(11, 11), Point(-1, -1));//均值滤波 用了11*11的双向算子。Point就用-1 -1
	imshow(output_title, dst);

	Mat gblur;
	GaussianBlur(src, gblur, Size(11, 11), 11, 11);//高斯模糊
	imshow("gaussian blur", gblur);

    //medianBlur(src, dst, 3);  高斯双边滤波
	bilateralFilter(src, dst, 15, 100, 5);
	namedWindow("BiBlur Filter Result", CV_WINDOW_AUTOSIZE);
	imshow("BiBlur Filter Result", dst);
    
    //用filter2D做通用滤波
    Mat resultImg;
	Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(dst, resultImg, -1, kernel, Point(-1, -1), 0);
	imshow("Final Result", resultImg);
    
    
	waitKey(0);
	return 0;
}
```

## 膨胀与腐蚀

### 理论说明：

​	图像形态学操作 – 基于形状的一系列图像处理操作的合集，主要是基于集合论基础上的形态学数学

​	形态学有四个基本操作：腐蚀、膨胀、开、闭

​	膨胀与腐蚀是图像处理中最常用的形态学操作手段，用在二值或者灰度图像上更有使用意义。

​	膨胀（二值图像中孤立的白点或者黑点会变小或者消失。）：跟卷积操作类似，假设有图像A和结构元素B，结构元素B在A上面移动，其中B定义其中心为锚点，计算B覆盖下A的最大像素值用来替换锚点的像素，其中B作为结构体可以是任意形状 

​	腐蚀（二值图像中孤立的白点或者黑点会变大）：腐蚀跟膨胀操作的过程类似，唯一不同的是以最小值替换锚点重叠下图像的像素值 

​	用途：通过膨胀腐蚀的方式“缩放”处理图像来消除大块的干扰。或者增强需要的局部图像元素。

### 相关API:

​	getStructuringElement(int shape, Size ksize, Point anchor) 

​		形状参数 (MORPH_RECT \MORPH_CROSS \MORPH_ELLIPSE) 

​		Size 必须是奇数 

​		锚点 默认是Point(-1, -1)意思就是中心像素 

​	dilate(src, dst, structureElement, Point(-1, -1), 1);//膨胀
    	erode(src, dst, structureElement);//腐蚀

### HIGHGUI图形化简单DEMO：

​	lTrackBar – createTrackbar(const String & trackbarname, const String winName,  int* value, int count, Trackbarcallback func, void* userdata=0) 

​	其中最中要的是 callback 函数功能。如果设置为NULL就是说只有值update，但是不会调用callback的函数。 

### 代码演示：

```c++
#include <opencv2/opencv.hpp> 
#include <iostream> 
using namespace cv;

Mat src, dst;
char OUTPUT_WIN[] = "output image";
int element_size = 3;//设立结构元素大小，这里用个变量，方便gui函数调整赋值
int max_size = 21;//gui设置值的上限
void CallBack_Demo(int, void*);
int main(int argc, char** argv) {
	
	src = imread("D:/vcprojects/images/test1.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);

	namedWindow(OUTPUT_WIN, CV_WINDOW_AUTOSIZE);
    
    //创建一个图形化的滑条，调整参数，会调整element_size的值
	createTrackbar("Element Size :", OUTPUT_WIN, &element_size, max_size, CallBack_Demo);
    
	CallBack_Demo(0, 0);//以调用函数的方式来处理，适合被gui操作

	waitKey(0);
	return 0;
}

void CallBack_Demo(int, void*) {
	int s = element_size * 2 + 1;//这个值必须是奇数
    
    //获取结构元素，用正方形的核，核的大小会随element_size变化
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
	// dilate(src, dst, structureElement, Point(-1, -1), 1);//膨胀
    erode(src, dst, structureElement);//腐蚀
	imshow(OUTPUT_WIN, dst);
	return;
}
```

## 形态学操作：开  闭：

​	二值图像中有实际意义的。

### 	开操作：

​		先腐蚀后膨胀  ，可以去掉小的对象，假设对象是前景色，背景是黑色 

### 	闭操作：

​		先膨胀后腐蚀 ，可以填充小的洞（fill hole），假设对象是前景色，背景是黑色 "

### 	形体学梯度：

​		膨胀减去腐蚀 ，又称为基本梯度（其它还包括-内部梯度、方向梯度，opencv不支持，但可以通过代码自我实现。） 

​		内部梯度：腐蚀减去原图

​		方向梯度：在X或者Y方向的梯度计算



### 	顶帽：

​		顶帽 是原图像与开操作之间的差值图像 

### 	黑帽：

​		黑帽是闭操作图像与源图像的差值图像

### 	相关API：

​	lmorphologyEx(src, dest, CV_MOP_BLACKHAT, kernel);

 - Mat src – 输入图像

 - Mat dest – 输出结果

- int OPT –   CV_MOP_OPEN/ CV_MOP_CLOSE/ CV_MOP_GRADIENT / CV_MOP_TOPHAT/ CV_MOP_BLACKHAT 形态学操作类型

-Mat kernel 结构元素

-int Iteration 迭代次数，默认是1

### 	代码演示：

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
int main(int argc, char** argv) {
	Mat src, dst;
	src = imread("D:/vcprojects/images/bin2.png");
	if (!src.data) {
		printf("could not load image...\n");
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);
	char output_title[] = "morphology demo";
	namedWindow(output_title, CV_WINDOW_AUTOSIZE);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
	morphologyEx(src, dst, CV_MOP_BLACKHAT, kernel);//一个函数根据参数不同实现上午不同的功能
	imshow(output_title, dst);

	waitKey(0);
	return 0;
}
```

## 形态学操作——提取水平与垂直线

### 原理方法：

​	方法有很多，这里只说其中一种。

​	图像形态学操作时候，可以通过自定义的结构元素实现结构元素对输入图像一些对象敏感、另外一些对象不敏感，这样就会让敏感的对象改变而不敏感的对象保留输出。通过使用两个最基本的形态学操作 – 膨胀与腐蚀，使用不同的结构元素实现对输入图像的操作、得到想要的结果。

- 膨胀，输出的像素值是结构元素覆盖下输入图像的最大像素值

- 腐蚀，输出的像素值是结构元素覆盖下输入图像的最小像素值

  提取水平或者垂直线的方式就是，假如提取水平线，则把垂直线给覆盖掉后再处理，步骤：

  ​	输入图像彩色图像 imread

  ​	转换为灰度图像 – cvtColor

  ​	转换为二值图像 – adaptiveThreshold

  ​	定义结构元素

  ​	开操作 （腐蚀+膨胀）提取 水平与垂直线

### 相关API：



### 代码演示：

```c++
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
int main(int argc, char** argv) {
	Mat src, dst;
	src = imread("D:/vcprojects/images/chars.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}

	char INPUT_WIN[] = "input image";
	char OUTPUT_WIN[] = "result image";
	namedWindow(INPUT_WIN, CV_WINDOW_AUTOSIZE);
	imshow(INPUT_WIN, src);

	Mat gray_src;
	cvtColor(src, gray_src, CV_BGR2GRAY);//转灰度
	imshow("gray image", gray_src);
	
	Mat binImg;
	adaptiveThreshold(~gray_src, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);//二值，注意可以用这两个阈值类型ADAPTIVE_THRESH_MEAN_C ， ADAPTIVE_THRESH_GAUSSIAN_C 
	imshow("binary image", binImg);

	// 水平结构元素
	Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
	// 垂直结构元素
	Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16), Point(-1, -1));
	// 矩形结构 用这种kernel，结合下面的处理步骤，可以在实际应用中消除图像的线的干扰和点的干扰。（车牌识别中去除车牌上的脏）
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

	Mat temp;
	erode(binImg, temp, kernel);//腐蚀 hline保留水平线  vline保留垂直线 类似在不同方向梯度处理了
	dilate(temp, dst, kernel);//膨胀 hline保留水平线  vline保留垂直线
	// morphologyEx(binImg, dst, CV_MOP_OPEN, vline);//上述两句代替这一句操作，可以更理解原理
	bitwise_not(dst, dst);//按位取反，看的更明显。还有or and 等等操作，在之前有说过
	//blur(dst, dst, Size(3, 3), Point(-1, -1));//均值滤波，结果更圆滑一点，一点小的输出技巧
	imshow("Final Result", dst);

	waitKey(0);
	return 0;
}
```

图像金字塔——上采样和降采样：

