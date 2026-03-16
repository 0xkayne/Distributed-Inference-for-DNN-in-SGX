/*
 * cpuid_sgx_probe.c — CPUID-level SGX/SGX2 feature detection
 *
 * Pure C, no external dependencies. Compile with: gcc -O2 -o cpuid_sgx_probe cpuid_sgx_probe.c
 *
 * Checks:
 *   1. CPUID 0x07:0 EBX bit 2  — SGX supported
 *   2. CPUID 0x12:0 EAX bit 0  — SGX1
 *   3. CPUID 0x12:0 EAX bit 1  — SGX2 (EDMM: EAUG/EMODT/EMODPR/EACCEPT)
 *   4. CPUID 0x12:0 EBX         — MISCSELECT (bit 0 = EXINFO)
 *   5. CPUID 0x12:1             — SECS.ATTRIBUTES valid bits
 *   6. CPUID 0x12:2+            — EPC section enumeration
 *   7. CPUID 0x40000000         — Hypervisor vendor string
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* ── CPUID helper ─────────────────────────────────────────────── */

static inline void cpuid_count(uint32_t leaf, uint32_t subleaf,
                               uint32_t *eax, uint32_t *ebx,
                               uint32_t *ecx, uint32_t *edx)
{
    __asm__ __volatile__("cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf));
}

/* ── Pretty-print helpers ─────────────────────────────────────── */

static const char *yes_no(int v) { return v ? "YES" : "NO"; }

static void print_separator(void)
{
    printf("────────────────────────────────────────────────────\n");
}

static void print_section(const char *title)
{
    printf("\n");
    print_separator();
    printf("  %s\n", title);
    print_separator();
}

/* ── Hypervisor detection ─────────────────────────────────────── */

static int detect_hypervisor(char *vendor_out, size_t len)
{
    uint32_t eax, ebx, ecx, edx;

    /* Check hypervisor present bit: CPUID 0x01 ECX bit 31 */
    cpuid_count(0x01, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1u << 31))) {
        snprintf(vendor_out, len, "(bare metal)");
        return 0;
    }

    /* CPUID 0x40000000: hypervisor vendor string in EBX:ECX:EDX */
    cpuid_count(0x40000000, 0, &eax, &ebx, &ecx, &edx);

    char vendor[13];
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &ecx, 4);
    memcpy(vendor + 8, &edx, 4);
    vendor[12] = '\0';

    snprintf(vendor_out, len, "%s", vendor);
    return 1;
}

/* ── SGX feature detection ────────────────────────────────────── */

static int check_sgx_support(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid_count(0x07, 0, &eax, &ebx, &ecx, &edx);
    return (ebx >> 2) & 1;  /* EBX bit 2 = SGX */
}

static void probe_sgx_capabilities(int *sgx1, int *sgx2, uint32_t *miscselect)
{
    uint32_t eax, ebx, ecx, edx;

    /* CPUID 0x12:0 — SGX capability enumeration */
    cpuid_count(0x12, 0, &eax, &ebx, &ecx, &edx);

    *sgx1 = (eax >> 0) & 1;   /* EAX bit 0 = SGX1 */
    *sgx2 = (eax >> 1) & 1;   /* EAX bit 1 = SGX2 (EDMM) */
    *miscselect = ebx;         /* Supported MISCSELECT bits */

    printf("  CPUID 0x12:0  EAX = 0x%08X\n", eax);
    printf("                EBX = 0x%08X (MISCSELECT)\n", ebx);
    printf("                ECX = 0x%08X\n", ecx);
    printf("                EDX = 0x%08X\n", edx);
    printf("\n");
    printf("  SGX1 (EAX bit 0):  %s\n", yes_no(*sgx1));
    printf("  SGX2 (EAX bit 1):  %s  ← EDMM: EAUG/EMODT/EMODPR/EACCEPT\n",
           yes_no(*sgx2));
    printf("  EXINFO (MISCSELECT bit 0): %s\n",
           yes_no(*miscselect & 1));
}

static void probe_sgx_attributes(void)
{
    uint32_t eax, ebx, ecx, edx;

    /* CPUID 0x12:1 — Valid SECS.ATTRIBUTES bits */
    cpuid_count(0x12, 1, &eax, &ebx, &ecx, &edx);

    uint64_t attrs_lo = ((uint64_t)ebx << 32) | eax;
    uint64_t attrs_hi = ((uint64_t)edx << 32) | ecx;

    printf("  CPUID 0x12:1  SECS.ATTRIBUTES valid bits:\n");
    printf("    low  = 0x%016lX\n", (unsigned long)attrs_lo);
    printf("    high = 0x%016lX  (XSAVE feature mask)\n", (unsigned long)attrs_hi);
    printf("\n");

    /* Decode key attribute bits */
    printf("    DEBUG (bit 1):       %s\n", yes_no(eax & (1u << 1)));
    printf("    MODE64BIT (bit 2):   %s\n", yes_no(eax & (1u << 2)));
    printf("    PROVISIONKEY (bit 4):%s\n", yes_no(eax & (1u << 4)));
    printf("    EINITTOKEN (bit 5):  %s\n", yes_no(eax & (1u << 5)));
    printf("    CET (bit 6):         %s\n", yes_no(eax & (1u << 6)));
    printf("    KSS (bit 7):         %s\n", yes_no(eax & (1u << 7)));
}

static int enumerate_epc_sections(void)
{
    uint32_t eax, ebx, ecx, edx;
    int section_count = 0;
    uint64_t total_bytes = 0;

    printf("  EPC Sections (CPUID 0x12:2+):\n\n");

    for (uint32_t sub = 2; sub < 10; sub++) {
        cpuid_count(0x12, sub, &eax, &ebx, &ecx, &edx);

        uint32_t type = eax & 0xF;
        if (type == 0)
            break;  /* No more sections */

        if (type == 1) {
            /* Valid EPC section */
            uint64_t base = ((uint64_t)(ebx & 0xFFFFF) << 32) | (eax & 0xFFFFF000);
            uint64_t size = ((uint64_t)(edx & 0xFFFFF) << 32) | (ecx & 0xFFFFF000);

            printf("    Section %d: base=0x%012lX  size=0x%012lX (%lu MB)\n",
                   section_count, (unsigned long)base, (unsigned long)size,
                   (unsigned long)(size >> 20));
            total_bytes += size;
            section_count++;
        }
    }

    if (section_count == 0) {
        printf("    (no EPC sections found)\n");
    } else {
        printf("\n    Total EPC: %lu MB (%.1f GB)\n",
               (unsigned long)(total_bytes >> 20),
               (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    return section_count;
}

/* ── FLC detection ────────────────────────────────────────────── */

static int check_flc_support(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid_count(0x07, 0, &eax, &ebx, &ecx, &edx);
    return (ecx >> 30) & 1;  /* ECX bit 30 = SGX_LC (FLC) */
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void)
{
    int sgx1 = 0, sgx2 = 0, flc = 0;
    uint32_t miscselect = 0;
    char hv_vendor[64] = {0};

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║      CPUID SGX/SGX2 Feature Probe               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    /* ── Hypervisor ──────────────────────────────── */
    print_section("Hypervisor Detection");
    int is_vm = detect_hypervisor(hv_vendor, sizeof(hv_vendor));
    printf("  Virtualized:  %s\n", yes_no(is_vm));
    printf("  Vendor:       %s\n", hv_vendor);
    if (is_vm) {
        printf("\n  ⚠  Running in a VM. SGX2 availability depends on\n");
        printf("     hypervisor passthrough configuration.\n");
    }

    /* ── Basic SGX support ───────────────────────── */
    print_section("SGX Support (CPUID 0x07:0)");
    int sgx_supported = check_sgx_support();
    flc = check_flc_support();
    printf("  SGX (EBX bit 2):     %s\n", yes_no(sgx_supported));
    printf("  SGX_LC / FLC (ECX bit 30): %s\n", yes_no(flc));
    printf("\n  NOTE: SGX_LC (Flexible Launch Control) is independent\n");
    printf("        of SGX2. FLC != SGX2.\n");

    if (!sgx_supported) {
        printf("\n  ✗ SGX not supported by this CPU (or not exposed by hypervisor).\n");
        printf("    Cannot proceed with SGX2 detection.\n\n");
        return 1;
    }

    /* ── SGX capability leaf ─────────────────────── */
    print_section("SGX Capabilities (CPUID 0x12:0)");
    probe_sgx_capabilities(&sgx1, &sgx2, &miscselect);

    /* ── SGX attributes ──────────────────────────── */
    print_section("SGX Attributes (CPUID 0x12:1)");
    probe_sgx_attributes();

    /* ── EPC enumeration ─────────────────────────── */
    print_section("EPC Sections (CPUID 0x12:2+)");
    int epc_count = enumerate_epc_sections();

    /* ── Summary ─────────────────────────────────── */
    print_section("Summary");

    printf("  %-30s %s\n", "SGX supported:", yes_no(sgx_supported));
    printf("  %-30s %s\n", "SGX1:", yes_no(sgx1));
    printf("  %-30s %s\n", "SGX2 (EDMM):", yes_no(sgx2));
    printf("  %-30s %s\n", "FLC (Launch Control):", yes_no(flc));
    printf("  %-30s %s\n", "EXINFO:", yes_no(miscselect & 1));
    printf("  %-30s %d section(s)\n", "EPC:", epc_count);
    printf("  %-30s %s\n", "Virtualized:", is_vm ? hv_vendor : "No");
    printf("\n");

    if (sgx2) {
        printf("  ✓ CPUID reports SGX2 support (EDMM instructions available).\n");
        printf("    EAUG, EMODT, EMODPR, EACCEPT should work.\n");
    } else {
        printf("  ✗ CPUID does NOT report SGX2.\n");
        if (is_vm) {
            printf("    Likely cause: hypervisor is not passing through SGX2.\n");
            printf("    The physical CPU (Ice Lake+) may support SGX2, but\n");
            printf("    the KVM/QEMU configuration does not expose it.\n");
        } else {
            printf("    This CPU does not support SGX2 EDMM instructions.\n");
        }
    }
    printf("\n");

    /* Exit code: 0 = SGX2 present, 1 = SGX2 absent, 2 = SGX absent */
    if (!sgx_supported) return 2;
    return sgx2 ? 0 : 1;
}
