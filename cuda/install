TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o indices_op.cu.o indices_op.cu.cc \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -w -DNDEBUG

g++ -std=c++11 -shared -o indices_op.so indices_op.cc \
indices_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

nvcc -std=c++11 -c -o hist_op.cu.o hist_op.cu.cc \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -w -DNDEBUG

g++ -std=c++11 -shared -o hist_op.so hist_op.cc \
hist_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

nvcc -std=c++11 -c -o gat_op.cu.o gat_op.cu.cc \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -w -DNDEBUG

g++ -std=c++11 -shared -o gat_op.so gat_op.cc \
gat_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

nvcc -std=c++11 -c -o hist1_op.cu.o hist1_op.cu.cc \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -w -DNDEBUG

g++ -std=c++11 -shared -o hist1_op.so hist1_op.cc \
hist1_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

nvcc -std=c++11 -c -o gat1_op.cu.o gat1_op.cu.cc \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -w -DNDEBUG

g++ -std=c++11 -shared -o gat1_op.so gat1_op.cc \
gat1_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}