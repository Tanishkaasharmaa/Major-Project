
# (query, expected_faq_id or None)

EVAL_SET = [

    # ============================================================
    # FAQ 0 — How do I apply for Wi-Fi access?
    # ============================================================
    ("how do I apply for wifi", 0),
    ("how can I get wifi access", 0),
    ("where do I apply for university wifi", 0),
    ("what is the process to register for wifi", 0),

    # ============================================================
    # FAQ 1 — How long does Wi-Fi approval usually take?
    # ============================================================
    ("how long does wifi approval take", 1),
    ("when will my wifi application get approved", 1),
    ("how many days does wifi approval usually take", 1),
    ("I applied for wifi, when should I expect approval", 1),

    # ============================================================
    # FAQ 2 — Wi-Fi application rejected
    # ============================================================
    ("my wifi application was rejected what should I do", 2),
    ("what do I do if my wifi request got rejected", 2),
    ("they rejected my wifi application", 2),
    ("my wifi request was turned down, how can I fix it", 2),

    # ============================================================
    # FAQ 3 — Verification pending
    # ============================================================
    ("what does verification pending mean", 3),
    ("my wifi application says verification pending", 3),
    ("why is my wifi verification still pending", 3),
    ("what happens when my application is stuck on pending verification", 3),

    # ============================================================
    # FAQ 4 — What is a MAC address?
    # ============================================================
    ("what is a mac address", 4),
    ("why do you need my mac address for wifi", 4),
    ("why do I have to submit a mac address", 4),
    ("what is the mac address used for", 4),

    # ============================================================
    # FAQ 5 — How to find MAC address
    # ============================================================
    ("how do I find my laptop mac address", 5),
    ("where can I find the mac address of my device", 5),
    ("how can I check my wifi adapter mac address", 5),
    ("where do I see the physical address on windows", 5),

    # ============================================================
    # FAQ 6 — Wi-Fi was working but suddenly stopped
    # ============================================================
    ("my wifi was working but suddenly stopped", 6),
    ("wifi was working yesterday but is not working now", 6),
    ("my laptop was connected to wifi and suddenly it stopped", 6),
    ("wifi suddenly stopped working on my device", 6),

    # ============================================================
    # FAQ 7 — Update MAC address after changing device
    # ============================================================
    ("I got a new laptop how do I update my mac address", 7),
    ("how can I change the registered mac address", 7),
    ("I changed my phone and need to update my wifi device", 7),
    ("I have a new device, how do I register its mac address", 7),

    # ============================================================
    # FAQ 8 — Application verified but Wi-Fi doesn't work
    # ============================================================
    ("my application is verified but wifi still doesn't work", 8),
    ("wifi is not working even though my application says verified", 8),
    ("my wifi application shows verified but I cannot connect", 8),
    ("I have been verified but I still don't have wifi access", 8),

    # ============================================================
    # FAQ 9 — Never received notification that Wi-Fi is ready
    # ============================================================
    ("I was never told that my wifi was ready", 9),
    ("how do I know if my wifi account has been created", 9),
    ("I didn't receive any notification for wifi activation", 9),
    ("my wifi may be ready but I didn't get any message", 9),

    # ============================================================
    # FAQ 10 — Computer Centre support hours
    # ============================================================
    ("what are the computer centre working hours", 10),
    ("when is the computer centre help desk open", 10),
    ("what time does the computer centre open and close", 10),
    ("can I visit the computer centre for help after five", 10),

    # ============================================================
    # FAQ 11 — Reset Wi-Fi password
    # ============================================================
    ("how do I reset my wifi password", 11),
    ("I forgot my wifi password how can I reset it", 11),
    ("where can I request a wifi password reset", 11),
    ("I need to change or reset my wifi password", 11),

    # ============================================================
    # FAQ 12 — Mobile number not verified for OTP
    # ============================================================
    ("my mobile number is not verified for otp", 12),
    ("I cannot receive the otp because my number is not verified", 12),
    ("what should I do if my phone number is not verified", 12),
    ("otp is not working because my mobile number is not registered", 12),

    # ============================================================
    # FAQ 13 — Wi-Fi in hostel and academic building
    # ============================================================
    ("can I use the same wifi login in hostel and academic building", 13),
    ("does one wifi account work in the hostel too", 13),
    ("can I use university wifi in different buildings", 13),
    ("will my wifi login work in both hostel and college buildings", 13),

    # ============================================================
    # FAQ 14 — Problem not solved
    # ============================================================
    ("what should I do if none of these solutions work", 14),
    ("my wifi problem is still not solved what can I do", 14),
    ("I tried everything but my problem is still there", 14),
    ("how do I raise a complaint if the issue continues", 14),

    # ============================================================
    # FAQ 15 — Who approves Wi-Fi applications?
    # ============================================================
    ("who approves the wifi application", 15),
    ("does the provost or chairman approve my wifi request", 15),
    ("who needs to approve my wifi application", 15),
    ("who approves wifi for hostel students and day scholars", 15),

    # ============================================================
    # FAQ 16 — More than one device
    # ============================================================
    ("can I connect two devices to one wifi account", 16),
    ("can I use my wifi account on both my phone and laptop", 16),
    ("can I register more than one device for wifi", 16),
    ("does one wifi account allow multiple devices", 16),

    # ============================================================
    # FAQ 17 — Cost of Wi-Fi
    # ============================================================
    ("is university wifi free", 17),
    ("do I have to pay for wifi access", 17),
    ("is there any fee for using campus wifi", 17),
    ("how much does wifi access cost", 17),

    # ============================================================
    # OUT-OF-SCOPE / NONE
    #
    # These are important because your system must learn when
    # NOT to confidently answer using an unrelated FAQ.
    # Many are based on the style of real issues in issues.xlsx.
    # ============================================================

    ("the wifi router is not working", None),
    ("there is no wifi in my hostel", None),
    ("the router near my room is not working", None),
    ("wifi is not available in my department", None),
    ("the internet is not working in the reading room", None),
    ("the wifi keeps disconnecting every few minutes", None),
    ("wifi signal is very weak in my room", None),
    ("the wifi router cable is disconnected", None),
    ("there is no internet connection in the computer centre", None),
    ("the wifi access point is not working", None),
    ("my Samarth login ID does not exist", None),
    ("tell me a joke", None),
]