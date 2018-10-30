## Catatan Pengembangan

Berikut adalah porting seluruh python-class STI menjadi fungsi-fungsi Matlab.Seluruh kode sumber Matlab sifatnya masih DRAFT dan perlu perbaikan/pengembangan lebih jauh.

### Hasil Python original

Berikut adalah screenshot running skip python (sebelum porting ke Matlab):
![HasilPython](https://raw.githubusercontent.com/mekatronik-achmadi/sti-wav/master/src/matlab/sti_py.png)

### Kendala Porting
Beberapa kendala porting dari Python3 ke Matlab:

* Matlab fokus ke prosedural, sedangkan Python fokus pada objek dan class
* Matlab sign variabel otomatis, sedangkan Python memperhatikan type
* Matlab menggunakan for-loop bersifat dasar, sedangkan Python seringkali lebih kompleks
* Matlab memulai array/vector dari 1, sedangkan Python dari 0 (butuh cek lebih lanjut)
* Matlab menangani array/vector dengan row,column, sedangkan Python (Numpy) menangani column,row (butuh cek lebih lanjut)

### Struktur Kode Sumber

* Skrip test: test.m
* Fungsi Utama: stiFromAudio.m
* Fungsi-Fungsi belum terdefinisi: ./unreferenced/

### Fungsi-Fungsi belum terdefisini
Berikut adalah daftar fungsi belum terdefinisi karena belum didapat fungsi yang menggantikan di Matlab beserta fungsi yang memanggilnya:

* my_firwin.m : octaveBandFilter.m
* my_psd.m : octaveBandSpectra.m
* my_cohere.m : octaveBandCoherence.m
* my_searchsorted.m : thirdOctaveRMS.m, thirdOctaveRootSum.m
* my_clip.m : sti.m
* my_masked_array.m : sti.m

### Last Thinking
Untuk running skrip Python ini, sebenarnya hanya butuh install paket Python3, Numpy. Matplotlib, dan Scipy.
Jika butuh software antarmuka lingkungan layaknya Matlab, bisa install Sypder.
Untuk instalasi di sistem Windows, telah tersedia distribusi Anaconda yang siap download dan instal.
Sehingga pada dasarnya, lebih mudah dijalankan di Python daripada porting ke Matlab
