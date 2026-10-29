1. Fig.6的统计结果位于 config_error_stats.csv 中。
2. Fig. 7 的统计结果位于 configuration_error_stats.csv 中。使用的是CCI（即最快模式下的结果）。
3. CV mode和CC mode的具体设置为
CV mode:
frequency: f = 1/((M+1)*(N/1000+5\tau))
duty cycle: 25%, phase offset: 0^\circ, 90^\circ, 180^\circ
设M=3, N=12, 5\tau=0.06s, frequency为f=1/(4*(12/1000+0.06))=3.47Hz，
CC mode:
frequency: f = 1/(M*(N/1000+0.02))
duty cycle: 33%, phase offset: 0^\circ, 33^\circ, 66^\circ
设M=3， N=3(最小的使用3个传感器)， 0.02s为每个周期的时间，频率为f=1/(3*(3/1000+0.02))=14.49Hz，（TDM的最快频率）
4. Config 1对应使用12个全部传感器，Config2对应使用顶点的8个传感器，Config3对应最小的使用3个传感器，具体为2-3-12。
5. sequence就是从起点到终点的手动连续移动。ground truth就是ZED2i+AprilTag。如果这里前文表述不一致，那么前文也需要修改，我们一定是使用ZED2i+AprilTag作为ground truth的。
6. 没有