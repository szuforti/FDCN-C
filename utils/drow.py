
import matplotlib.pyplot as plt

def mscatter(x, y, m, c, **kw):
    """
    画不同形状的图
    :param x: 数据x，x[:,0]
    :param y: 数据y, x[:,1]
    :param m: 形状分类
    :param c: 预测值y,用来分类颜色
    :param kw: 其它参数
    :return:
    """
    import matplotlib.markers as mmarkers
    ax = plt.gca()
    sc = ax.scatter(x, y, c=c, **kw)
    m = list(map(lambda x: m[x], c))
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc