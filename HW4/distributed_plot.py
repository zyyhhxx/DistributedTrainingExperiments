import matplotlib.pyplot as plt

e_base = 483.483
p_base = 105.485831111
plt.plot([1, 210.971662222/p_base, 421.943324444/p_base, 632.914986667/p_base], label="Paleo predictions")
plt.plot([1, 493.265/e_base, 375.792/e_base, 293.830/e_base], label="experiments")
plt.xticks([0,1,2,3], [1,2,4,8])
plt.ylabel("average throughput")
plt.xlabel("number of GPUs")
plt.legend()
plt.savefig("3.1Ai.png")
print("Experiment plot is successfully generated")
plt.clf()

e_base = 103.416
p_base = 772.3650638041578
plt.plot([1, 397.4210354942896/p_base, 193.52277791974342/p_base, 100.35844097720052/p_base], label="Paleo predictions")
plt.plot([1, 101.365/e_base, 133.052/e_base, 127.246/e_base], label="experiments")
plt.xticks([0,1,2,3], [1,2,4,8])
plt.ylabel("runtime")
plt.xlabel("number of GPUs")
plt.legend()
plt.savefig("3.1Aii.png")
print("Experiment plot is successfully generated")
plt.clf()

e_base = 0.259
p_base = 105.485831111
plt.plot([1, 397.4210354942896/p_base*2, 193.52277791974342/p_base*4, 100.35844097720052/p_base*8], label="Paleo predictions")
plt.plot([1, 0.507/e_base, 1.331/e_base, 3.403/e_base], label="experiments")
plt.xticks([0,1,2,3], [1,2,4,8])
plt.ylabel("cost")
plt.xlabel("number of GPUs")
plt.legend()
plt.savefig("3.1Aiii.png")
print("Experiment plot is successfully generated")
plt.clf()

e_base = 483.483
e_tp = [1, 493.265/e_base, 375.792/e_base, 293.830/e_base]
p_base = 105.485831111
p_tp = [1, 210.971662222/p_base, 421.943324444/p_base, 632.914986667/p_base]
plt.plot([p_tp[i]/e_tp[i] for i in range(4)])
plt.xticks([0,1,2,3], [1,2,4,8])
plt.ylabel("percentage error")
plt.xlabel("number of GPUs")
plt.savefig("3.1B.png")
print("Experiment plot is successfully generated")
plt.clf()