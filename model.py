import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class NewsDetector:
    def __init__(self, dataset_path="data/news.csv"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.total_samples = 0
        self._ensure_dataset_exists()
        self._train_model()

    def _ensure_dataset_exists(self):
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rows = [
            # =========================================================================
            # 1. LEGITIMATE / RELIABLE / NON-SPAM (Label = 1)
            # =========================================================================
            # Conversational & Personal
            {"text": "Hey, are you free this evening? Let us meet at the coffee shop around 6 PM.", "label": 1},
            {"text": "Can you please send me the recipe for the chocolate cake you made last week?", "label": 1},
            {"text": "I had a wonderful time at the park today with my family. The weather was sunny and pleasant.", "label": 1},
            {"text": "Let us catch up over lunch on Friday if your schedule permits.", "label": 1},
            {"text": "Thanks for helping me move the furniture yesterday, I really appreciate your support.", "label": 1},
            {"text": "Happy birthday! Wishing you a fantastic year filled with health, joy, and success.", "label": 1},
            {"text": "Do you know what time the grocery store closes tonight? I need to pick up a few ingredients.", "label": 1},
            {"text": "I finished reading the novel you recommended and really enjoyed the character development.", "label": 1},
            {"text": "The weather is very pleasant today, let us go out for a coffee.", "label": 1},
            {"text": "Can you send the project presentation by 4 PM?", "label": 1},
            {"text": "Hello Naveen, can we review the presentation slides together before the meeting tomorrow?", "label": 1},
            {"text": "I just bought a new laptop today and it works very smoothly.", "label": 1},
            {"text": "Hey mom, I reached the hostel safely. Will call you after dinner.", "label": 1},
            {"text": "Did you watch the football match last night? What an amazing comeback in the final minutes!", "label": 1},
            {"text": "Please bring the house keys when you come back from office.", "label": 1},
            {"text": "I am stuck in heavy traffic near MG Road, will be late by 20 minutes.", "label": 1},
            {"text": "Can you share the lecture notes from yesterday's chemistry class?", "label": 1},
            {"text": "Great job on the presentation today, the client was really impressed with your analysis.", "label": 1},
            {"text": "Are we still meeting at the library at 3 PM to work on our group assignment?", "label": 1},
            {"text": "Don't forget to buy milk and bread on your way home.", "label": 1},
            {"text": "Thanks for the dinner invitation last night, we had a fantastic time!", "label": 1},
            {"text": "Could you please check if I left my jacket at your apartment?", "label": 1},

            # Workplace & Corporate Communication
            {"text": "Please find attached the quarterly financial report for review before our team meeting tomorrow morning.", "label": 1},
            {"text": "Reminder: The team standup is scheduled for 10:00 AM in Conference Room B.", "label": 1},
            {"text": "The team meeting has been moved to 10 AM on Monday.", "label": 1},
            {"text": "Thank you for submitting your project proposal. The review committee will provide feedback by next Tuesday.", "label": 1},
            {"text": "The marketing department completed the customer survey and shared the aggregate summary with team leads.", "label": 1},
            {"text": "Here are the meeting notes and action items from today's client sync.", "label": 1},
            {"text": "Please submit your timesheets by 5 PM today for payroll processing.", "label": 1},
            {"text": "Our engineering team scheduled server maintenance for midnight Sunday to apply security patches.", "label": 1},
            {"text": "Welcome to the team! Your onboarding session is scheduled for Monday at 9:30 AM.", "label": 1},
            {"text": "Hi team, please review the pull request on GitHub and leave your code review comments by EOD.", "label": 1},
            {"text": "The client requested a minor modification in contract clause 4. Please update the draft agreement.", "label": 1},
            {"text": "Annual performance evaluations will begin next week. Please complete your self-assessment form in HR portal.", "label": 1},
            {"text": "Attached is the monthly invoice for the software subscription renewal for Q3.", "label": 1},
            {"text": "Please make sure to update your JIRA tickets before the sprint planning meeting tomorrow.", "label": 1},
            {"text": "The server logs indicate that the API response time has improved by 25% after the database indexing update.", "label": 1},
            {"text": "HR announcement: The office will remain closed on Friday for the national holiday.", "label": 1},
            {"text": "Could you please send me the contact details of the vendor we hired for the event?", "label": 1},

            # Transactional & Service Notifications
            {"text": "Your order #12345 has been processed and will arrive by Friday. Thank you for shopping with us.", "label": 1},
            {"text": "Your appointment with Dr. Williams has been confirmed for Thursday at 2:30 PM.", "label": 1},
            {"text": "Your flight check-in is now open. Seat 14A has been assigned for your flight to Chicago.", "label": 1},
            {"text": "Your package has been delivered to your front porch. Have a great day.", "label": 1},
            {"text": "Your monthly utility statement is ready to view online. Your auto-pay is scheduled for the 15th.", "label": 1},
            {"text": "Your car service appointment is scheduled for tomorrow morning at 8:00 AM.", "label": 1},
            {"text": "The train arrives at platform 3 at 5:45 PM.", "label": 1},
            {"text": "Your Amazon order #402-8912301 has been dispatched. Track your delivery on the Amazon app.", "label": 1},
            {"text": "Your Uber ride with driver Suresh is arriving in 3 minutes. Vehicle number KA-01-MJ-4820.", "label": 1},
            {"text": "Your Zomato order from Spice Garden is out for delivery. Delivery executive contact: 9876501234.", "label": 1},
            {"text": "Your ticket booking for movie at PVR Cinemas is confirmed. Booking ID: PVR89201.", "label": 1},
            {"text": "Your Flipkart shipment is out for delivery today. Use OTP 4812 to accept parcel from delivery agent.", "label": 1},
            {"text": "Your diagnostic lab test report is ready. Download your digital PDF from the official Apollo portal.", "label": 1},
            {"text": "Your LPG gas cylinder booking reference #89120 has been registered. Expected delivery in 2 business days.", "label": 1},

            # Genuine Banking, OTPs & Financial Alerts
            {"text": "OTP is 492018 for your transaction of Rs 1,200 at HDFC Bank card ending 4012. Valid for 10 mins. Do not share with anyone.", "label": 1},
            {"text": "Dear customer, your salary of Rs 85,000 has been credited to your ICICI bank account XX9012 on 01-Sep-2026.", "label": 1},
            {"text": "Your SBI Credit Card statement for August 2026 is generated. Total amount due: Rs 4,320. Due date: 20-Sep-2026.", "label": 1},
            {"text": "Alert: You spent Rs 350 at Starbucks using Paytm UPI. Available balance: Rs 14,250.", "label": 1},
            {"text": "Your Axis Bank e-statement for account ending 5541 has been sent to your registered email address.", "label": 1},
            {"text": "Dear user, 1000 INR debited from your account via ATM cash withdrawal at MG Road branch.", "label": 1},
            {"text": "Your mutual fund SIP of Rs 5,00,000 has been successfully executed for Zerodha account ending 9120.", "label": 1},
            {"text": "Your loan EMI of Rs 12,450 for vehicle loan has been auto-debited from your bank account.", "label": 1},
            {"text": "Dear customer, your fixed deposit of Rs 1,00,000 has matured and credited to your savings account.", "label": 1},

            # Genuine Brand Offers & Discounts
            {"text": "Domino's Special Offer: Get 20% off on medium pizzas using code PIZZA20 on our official Domino's app today.", "label": 1},
            {"text": "Myntra End of Season Sale: Flat 50% discount on top fashion brands starting midnight. Visit official Myntra app.", "label": 1},
            {"text": "Book your flight ticket on MakeMyTrip and get flat Rs 500 cashback with HDFC bank cards.", "label": 1},
            {"text": "Flat 10% instant discount on electronics at Croma physical stores this weekend.", "label": 1},
            {"text": "Book movie tickets on BookMyShow and get 1-for-1 offer using ICICI debit card on weekends.", "label": 1},
            {"text": "Enjoy 15% discount on your next ride with Ola using coupon code RIDE15.", "label": 1},

            # Verified News, Science & Public Information
            {"text": "Local authorities opened a new public library in the downtown area with over 50,000 books and free internet access.", "label": 1},
            {"text": "Apple held its annual developer conference where they introduced new software features for iPhone and Mac users.", "label": 1},
            {"text": "The home team won the championship game last night with a final score of 3 to 1 in overtime.", "label": 1},
            {"text": "Tomorrow will be partly cloudy with temperatures reaching a high of 75 degrees and light breeze.", "label": 1},
            {"text": "The city transit authority expanded bus routes to improve connectivity across residential suburbs.", "label": 1},
            {"text": "The community farmers market will feature fresh organic produce and local crafts every Saturday morning.", "label": 1},
            {"text": "Sony announced its next-generation gaming console update with improved graphics and faster loading speeds.", "label": 1},
            {"text": "The municipal water board reported that tap water quality tests met all state safety standards this quarter.", "label": 1},
            {"text": "According to a study published in the New England Journal of Medicine, researchers found that regular cardiovascular exercise reduces heart disease risk by 32 percent across diverse age groups.", "label": 1},
            {"text": "The World Health Organization confirmed that global polio vaccination campaigns have reduced wild poliovirus cases by over 99 percent since 1988.", "label": 1},
            {"text": "Clinical trial results published in The Lancet demonstrate that the new mRNA malaria vaccine achieved 77 percent efficacy in pediatric trials across three West African countries.", "label": 1},
            {"text": "The Federal Reserve announced a 25 basis point interest rate adjustment following the Federal Open Market Committee meeting, citing moderation in core inflation figures.", "label": 1},
            {"text": "NASA scientists announced the discovery of a terrestrial exoplanet in the habitable zone of a nearby star system after analyzing spectrographic data from the James Webb Space Telescope.", "label": 1},
            {"text": "The Environmental Protection Agency released its annual emissions report indicating a 3.2 percent decline in nationwide greenhouse gas emissions.", "label": 1},
            {"text": "According to the Bureau of Labor Statistics, non-farm payroll employment increased by 216,000 jobs in December, while the national unemployment rate held steady at 3.7 percent.", "label": 1},
            {"text": "European Space Agency officials confirmed that the automated cargo resupply spacecraft successfully docked with the International Space Station.", "label": 1},

            # =========================================================================
            # 2. SPAM, SCAM, FRAUD & PHISHING (Label = 0)
            # =========================================================================
            # Prize, Lottery & Lucky Draw Scams
            {"text": "Congratulations! You have won ₹10,00,000 in our lucky draw. Click here to claim your prize now!", "label": 0},
            {"text": "CONGRATULATIONS! You have been selected as the official lucky winner of $1,000,000 in our international lottery! Transfer $250 processing fee immediately to claim your prize now.", "label": 0},
            {"text": "Congratulations you won a free iPhone 15 Pro Max! Click this link to confirm your delivery address and pay only $10 shipping fee right now.", "label": 0},
            {"text": "You are chosen for a free gift card worth $500. Click here to claim your reward.", "label": 0},
            {"text": "URGENT: Win ₹25,00,000 cash bonus now. Click link to claim.", "label": 0},
            {"text": "Congratulations user! You were randomly selected for a $1,000 Walmart gift card. Complete the survey and provide your credit card to verify eligibility.", "label": 0},
            {"text": "You won! Send ₹10,00,000 payment to unlock your free phone delivery to your address.", "label": 0},
            {"text": "CLAIM YOUR FREE REWARD! You have 1 unclaimed $500 Amazon gift voucher. Click http://claim-reward-now.xyz to redeem before midnight!", "label": 0},
            {"text": "Lucky draw alert: You won 50,00,000 INR from KBC jackpot! Call WhatsApp manager immediately at 9182391203 to claim.", "label": 0},
            {"text": "You have been selected as our top shopper of the month! Click here http://free-gift-shop.net to claim your free laptop immediately.", "label": 0},
            {"text": "CONGRATS! You won a $250 voucher from Target. Simply complete this 1 minute survey and enter your credit card info for shipping: http://target-survey.org", "label": 0},
            {"text": "Dear user, you won 1st prize in Spin & Win contest! Transfer Rs 500 registration charge to claim Rs 50,000 cash in account.", "label": 0},
            {"text": "WINNER ANNOUNCEMENT: Your mobile number won 25 Lakhs cash in Kaun Banega Crorepati mega draw. Contact manager Rahul on WhatsApp.", "label": 0},
            {"text": "Instant reward: Spin the wheel and win a guaranteed Samsung Galaxy S24! Claim your prize now at http://spin-win-reward.biz", "label": 0},
            {"text": "You have 1 pending reward points worth Rs 4,500 expiring today. Redeem cash directly to your bank account at http://rewards-redeem-online.com", "label": 0},

            # Banking Phishing & Credential Theft Scams
            {"text": "URGENT NOTICE: Your bank account has been flagged for suspicious activity. Click here right now and verify your online banking password and OTP to prevent immediate account termination.", "label": 0},
            {"text": "Dear customer, your electricity power will be disconnected tonight. Call officer at 9876543210.", "label": 0},
            {"text": "Your SBI Bank account is blocked due to KYC. Click link to update PAN card immediately.", "label": 0},
            {"text": "Dear user, your SIM card will be deactivated today. Call customer care immediately.", "label": 0},
            {"text": "FINAL WARNING: Your electric utility service will be permanently disconnected within 1 hour unless you immediately send payment via Bitcoin voucher.", "label": 0},
            {"text": "Pre-approved loan of ₹5,00,000 approved at 0% interest. Click here to disburse immediately.", "label": 0},
            {"text": "You received a $5,00,000 federal grant approved by the treasury. Send your full social security number and $100 processing charge to receive wire transfer.", "label": 0},
            {"text": "Lucky prize alert: You won ₹5,00,000 cash reward. Send your PAN card and bank details with ₹2,500 security deposit to release payment.", "label": 0},
            {"text": "Urgent package delivery pending: We could not deliver your parcel due to invalid address. Click here to confirm personal details and pay redelivery fee.", "label": 0},
            {"text": "Inheritance fund notification: You are named as sole beneficiary for 2.5 million dollars. Send your passport copy and release fee.", "label": 0},
            {"text": "HDFC Security Warning: Unauthorized login attempt detected from Russia. Click http://hdfc-security-fix.com to change your password immediately.", "label": 0},
            {"text": "Dear ICICI user, your netbanking access is suspended. Update your Aadhaar and PAN card at http://icici-verify-kyc.net within 24 hours.", "label": 0},
            {"text": "ALERT: Your Netflix subscription payment failed! Update your credit card details immediately at http://netflix-billing-update.com to avoid account deactivation.", "label": 0},
            {"text": "PayPal Alert: Your account has been restricted due to unauthorized transactions. Verify your identity now at http://paypal-security-center.info", "label": 0},
            {"text": "TRAI Notice: Your mobile number will be blocked within 2 hours due to illegal activities. Call TRAI officer immediately at 9182390123.", "label": 0},
            {"text": "India Post Alert: Your parcel #IN90123 is held at customs due to unpaid tax of Rs 48. Pay now at http://indiapost-redelivery.org to avoid return.", "label": 0},
            {"text": "Axis Bank Security: Suspicious transaction of Rs 45,000 detected on your credit card. If not done by you, immediately verify at http://axis-verify-login.org", "label": 0},
            {"text": "Income Tax Refund: You are eligible for tax refund of Rs 15,480. Click http://incometax-refund-claim.gov.in-verify.xyz to enter bank account details.", "label": 0},
            {"text": "FedEx Notice: Package delivery failed due to incorrect house number. Confirm address and pay $1.99 redelivery fee at http://fedex-parcel-track.info", "label": 0},
            {"text": "Bank of Baroda Alert: Your debit card is blocked. Download BOB World APK file from link http://bob-mobile-app.xyz to unblock.", "label": 0},
            {"text": "Apple ID Security: Your iCloud account has been locked. Verify your password and security questions at http://apple-id-verify-auth.com", "label": 0},
            {"text": "Challan Alert: Traffic police issued e-challan of Rs 2,000 for your vehicle. Pay immediately at http://echallan-parivahan-pay.xyz to avoid court summons.", "label": 0},

            # Fake Jobs, Get-Rich Schemes & Crypto Fraud
            {"text": "Limited-time investment opportunity! Double your money in 7 days. Join now!", "label": 0},
            {"text": "Exclusive opportunity: Make $15,000 per week working only 20 minutes a day from your phone! Send ₹5,000 registration fee to activate your automated crypto bot.", "label": 0},
            {"text": "Earn ₹5,000 daily working 1 hour from home. No experience needed. Join Telegram.", "label": 0},
            {"text": "Part time job: Earn 3000 to 5000 per day from mobile. Contact WhatsApp.", "label": 0},
            {"text": "Send 1000 rupees to receive 10000 rupees tomorrow guaranteed.", "label": 0},
            {"text": "Double your cryptocurrency in 24 hours! Send 0.1 BTC to the designated smart contract address and receive 0.2 BTC back instantly guaranteed!", "label": 0},
            {"text": "Guaranteed binary options trading profit: Deposit $200 today and withdraw $3,000 daily with our automated algorithm.", "label": 0},
            {"text": "Amazon hiring online rating staff! Work 30 mins daily from home and earn 5000 Rs per day. No skills required. WhatsApp 9988776655 now.", "label": 0},
            {"text": "Invest 5000 INR in our AI automated trading app and get 50,000 INR guaranteed returns in 48 hours! Limited slots left.", "label": 0},
            {"text": "YouTube like and subscribe job! Earn Rs 50 per video liked. Daily payout up to Rs 4000. Join Telegram channel http://t.me/earnmoneyfast", "label": 0},
            {"text": "Data entry work from home: Earn 25,000 per month typing simple captcha. No experience required. Pay refundable deposit of Rs 1,500 to get work materials.", "label": 0},
            {"text": "Stock market insider tips guaranteed 100% upper circuit stocks daily! Join our exclusive VIP WhatsApp group for 10x returns: http://chat.whatsapp.com/inv123", "label": 0},
            {"text": "Earn 500 USD per day working from home using Google search engine rating. Click http://google-search-jobs.online to apply now.", "label": 0},
            {"text": "Forex trading signals 99% accuracy! Turn $100 into $5,000 in 3 days. Join our VIP Telegram channel today.", "label": 0},
            {"text": "Online hotel rating job: Review luxury hotels on Google Maps and earn 3,000 Rs daily. Contact HR on Telegram @hoteljobs2026", "label": 0},

            # Clickbait & Fabricated Sensationalism
            {"text": "SHOCKING: You will never believe what doctors just admitted! This simple kitchen spice cures diabetes and cancer in 48 hours and Big Pharma is desperately trying to hide the secret from you!", "label": 0},
            {"text": "MUST READ: Famous celebrity caught red-handed in horrifying scandal that the mainstream media is desperately trying to bury from the public!", "label": 0},
            {"text": "VIRAL: Woman lost 45 pounds in just 4 days without exercise using this one weird ancient trick that nutritionists want banned immediately!", "label": 0},
            {"text": "EXCLUSIVE: Secret leaked video exposes what really happened behind closed doors at the summit! You will be horrified by the truth!", "label": 0},
            {"text": "BREAKING: Scientists reveal that eating this common everyday fruit is secretly destroying your liver and aging your skin by 20 years overnight!", "label": 0},
            {"text": "UNBELIEVABLE: Miracle water drops restore 20/20 vision in 3 days! Eye surgeons hate this man for sharing the forbidden recipe!", "label": 0},
            {"text": "WARNING: Throw away your microwave immediately! New whistleblower evidence shows it emits deadly cosmic radiation that alters human DNA!", "label": 0},
            {"text": "TOP SECRET: Billionaires are building underground bunkers because a hidden asteroid is scheduled to collide with Earth next Tuesday!", "label": 0},
            {"text": "EXCLUSIVE: Declassified military files prove that the moon landings were completely filmed on a Hollywood sound stage to trick rival nations, and secret elites are covering up the truth!", "label": 0},
            {"text": "Chemtrails revealed: Government planes are secretly spraying mind-control chemicals and synthetic pathogens over residential neighborhoods to reduce population!", "label": 0},
            {"text": "SHOCKING PROOF: 5G cell towers are transmitting subliminal biometric frequencies designed to manipulate citizen behavior and disable immune systems!", "label": 0},
            {"text": "The World Economic Forum is executing a covert master plan to ban all cash currency and replace it with microchips implanted under citizens' skin!", "label": 0},
            {"text": "Ancient pyramid texts deciphered by independent researchers prove that human civilization was engineered by reptilian extraterrestrials 5,000 years ago!", "label": 0},
            {"text": "BOMBSHELL REPORT: All weather satellites are completely fake! Earth is actually surrounded by a giant glass dome controlled by secret elites!", "label": 0},
            {"text": "Whistleblower inside the government exposes secret military weather machines causing all earthquakes and hurricanes across the globe!", "label": 0},
            {"text": "EXPOSED: Leading world leaders have already been replaced by biological clones operating from underground cloned facilities!", "label": 0},
            {"text": "Drinking pure chlorine dioxide solution 3 times a day is proven to cure autism, arthritis, and all viral infections according to suppressed holistic healers!", "label": 0},
            {"text": "Wearing face masks causes carbon dioxide poisoning and permanent brain damage within 10 minutes according to holistic freedom advocates!", "label": 0},
        ]

        df = pd.DataFrame(sample_rows)
        df.to_csv(self.dataset_path, index=False)
        self.df = df
        self.total_samples = len(df)

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " url_token ", text)
        text = re.sub(r"[\$\u20b9\u20ac\u00a3]", " currency_token ", text)
        text = re.sub(r"\b\d+\b", " num_token ", text)
        text = re.sub(r"[^\w\s_]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _train_model(self):
        if self.dataset_path.exists():
            self.df = pd.read_csv(self.dataset_path)
        else:
            self._ensure_dataset_exists()

        self.total_samples = len(self.df)
        df = self.df.copy()
        df["cleaned"] = df["text"].apply(self.clean_text)
        df = df.dropna(subset=["cleaned"])

        X = df["cleaned"]
        y = df["label"].astype(int)

        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        self.model = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        transformer_list=[
                            (
                                "word_tfidf",
                                TfidfVectorizer(
                                    ngram_range=(1, 3),
                                    min_df=1,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char_tfidf",
                                TfidfVectorizer(
                                    ngram_range=(3, 5),
                                    analyzer="char_wb",
                                    min_df=1,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=1.2,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ),
            ]
        )

        self.model.fit(train_X, train_y)
        predictions = self.model.predict(test_X)

        self.metrics = {
            "accuracy": round(float(accuracy_score(test_y, predictions)), 3),
            "precision": round(float(precision_score(test_y, predictions, zero_division=0)), 3),
            "recall": round(float(recall_score(test_y, predictions, zero_division=0)), 3),
            "f1": round(float(f1_score(test_y, predictions, zero_division=0)), 3),
        }

    def detect_signals(self, text):
        raw_text = str(text)
        lower = raw_text.lower()

        reliable_sources = [
            "according to", "officials said", "spokesperson", "study published in", "researchers found",
            "journal of", "published in", "confirmed by", "data shows", "statistics show", "research indicates",
            "department of", "agency announced", "organization reported", "expert says", "university",
            "reuters", "associated press", "world health organization", "centers for disease control",
            "federal reserve", "bureau of labor", "supreme court", "national oceanic", "geological survey",
            "peer-reviewed", "clinical trial", "meta-analysis", "audits confirmed", "nasa scientists",
            "environmental protection agency"
        ]

        clickbait_terms = [
            "must read", "must watch", "you will never believe", "you won't believe", "shocking", "unbelievable",
            "mind blown", "blow your mind", "doctors hate", "miracle cure", "one weird trick", "cures all",
            "what happened next", "exclusive leaked", "banned immediately", "this one trick", "viral video",
            "top secret", "insane discovery", "mind blowing", "bombshell", "hidden truth", "forbidden recipe",
            "caught red-handed", "trying to bury"
        ]

        emotional_terms = [
            "horrifying", "horrified", "furious", "terrified", "terrifying", "outrage", "deadly poison",
            "apocalypse", "nightmare", "evil plot", "atrocity", "betrayal", "disgraceful", "shameful",
            "mass panic", "hysteria", "scandal that will"
        ]

        conspiracy_terms = [
            "declassified", "moon landing", "moon landings", "sound stage", "hollywood sound stage",
            "5g cell towers", "5g towers", "chemtrails", "synthetic pathogens", "depopulation", "deep state",
            "new world order", "illuminati", "secret elite", "secret elites", "microchip", "biological clones",
            "cloned facilities", "reptilian", "weather machine", "weather machines", "mainstream media is desperately trying",
            "mainstream media is covering", "suppressed cure", "forbidden cure", "alien coverup", "flat earth",
            "glass dome", "chlorine dioxide", "big pharma is desperately trying"
        ]

        p_scam_heuristic = r"lucky\s*draw|lottery|won\s+.*(prize|lakh|crore|\$|\u20b9|cash|reward|iphone|gift)|claim\s+.*(prize|reward|cash|bonus)|double\s+.*(money|crypto|bitcoin)|part\s*time\s*job.*(earn|daily)|blocked.*(kyc|pan|verify)|disconnected.*(tonight|bill)"

        signals = []

        # --- Signal 1: Source Attribution ---
        source_matches = [t for t in reliable_sources if t in lower]
        if len(source_matches) >= 1:
            signals.append({
                "name": "Source Attribution",
                "status": "Verified",
                "severity": "safe",
                "detail": f"Contains reference to accredited source '{source_matches[0]}'.",
            })
        else:
            signals.append({
                "name": "Source Attribution",
                "status": "Direct / Clean",
                "severity": "safe",
                "detail": "Normal direct communication without external citation dependencies.",
            })

        # --- Signal 2: Clickbait & Sensationalism ---
        clickbait_matches = [t for t in clickbait_terms if t in lower]
        if len(clickbait_matches) >= 1:
            signals.append({
                "name": "Clickbait & Sensationalism",
                "status": "Detected",
                "severity": "warning",
                "detail": f"Found clickbait / sensational phrasing: \"{clickbait_matches[0]}\".",
            })
        else:
            signals.append({
                "name": "Clickbait & Sensationalism",
                "status": "Clear",
                "severity": "safe",
                "detail": "No sensationalist or clickbait tropes detected.",
            })

        # --- Signal 3: Emotional Manipulation ---
        emotional_matches = [t for t in emotional_terms if t in lower]
        if len(emotional_matches) >= 1:
            signals.append({
                "name": "Emotional Manipulation",
                "status": "Detected",
                "severity": "warning",
                "detail": f"Contains emotionally charged wording: \"{emotional_matches[0]}\".",
            })
        else:
            signals.append({
                "name": "Emotional Manipulation",
                "status": "Neutral Tone",
                "severity": "safe",
                "detail": "Objective and balanced communicative tone.",
            })

        # --- Signal 4: Conspiracy Tropes ---
        conspiracy_matches = [t for t in conspiracy_terms if t in lower]
        if len(conspiracy_matches) >= 1:
            signals.append({
                "name": "Conspiracy Tropes",
                "status": "Detected",
                "severity": "danger",
                "detail": f"Identified conspiracy trope: \"{conspiracy_matches[0]}\".",
            })
        else:
            signals.append({
                "name": "Conspiracy Tropes",
                "status": "Clear",
                "severity": "safe",
                "detail": "No conspiracy theory patterns or debunked tropes found.",
            })

        # --- Signal 5: Spam / Fraud Signature ---
        if re.search(p_scam_heuristic, lower):
            signals.append({
                "name": "Spam / Fraud Signature",
                "status": "Detected",
                "severity": "warning",
                "detail": "Matches commercial offer, prize, or urgency keywords evaluated by ML classifier.",
            })
        else:
            signals.append({
                "name": "Spam / Fraud Signature",
                "status": "Clear",
                "severity": "safe",
                "detail": "No commercial fraud, lottery, or phishing patterns identified.",
            })

        # --- Signal 6: Stylometry & Quality ---
        caps_count = sum(1 for ch in raw_text if ch.isupper())
        total_letters = sum(1 for ch in raw_text if ch.isalpha())
        caps_ratio = (caps_count / total_letters) if total_letters > 0 else 0
        exclamation_count = raw_text.count("!")

        if caps_ratio > 0.40 or exclamation_count >= 3:
            signals.append({
                "name": "Stylometry & Quality",
                "status": "Elevated Emphasis",
                "severity": "warning",
                "detail": f"Contains elevated emphasis formatting ({round(caps_ratio*100)}% capitalization).",
            })
        else:
            signals.append({
                "name": "Stylometry & Quality",
                "status": "Standard Quality",
                "severity": "safe",
                "detail": "Standard sentence structure, punctuation, and casing observed.",
            })

        return signals

    def predict_article(self, text):
        clean_t = self.clean_text(text)
        if not clean_t:
            raise ValueError("Text is empty after cleaning.")

        # 1. PURE ML CLASSIFIER PROBABILITY (CALIBRATED LINEAR SVC / LOGISTIC REGRESSION)
        probabilities = self.model.predict_proba([clean_t])[0]
        classes = list(self.model.classes_)
        if 1 in classes:
            reliability_score = float(probabilities[classes.index(1)])
        else:
            reliability_score = float(probabilities[0])

        # Clamp score between 0.01 and 0.99
        reliability_score = max(0.01, min(0.99, reliability_score))
        misleading_score = 1.0 - reliability_score

        # 2. GENERATE INFORMATIVE UI SIGNALS (Metadata only, no score overriding)
        signals = self.detect_signals(text)

        # 3. VERDICT & CONFIDENCE ESTIMATION
        if reliability_score >= 0.50:
            label = "Reliable"
            confidence = round(reliability_score * 100, 1)
        else:
            label = "Misleading"
            confidence = round(misleading_score * 100, 1)

        result = {
            "label": label,
            "confidence": confidence,
            "reliability": round(reliability_score, 3),
            "reliable_score": round(reliability_score * 100, 1),
            "misleading_score": round(misleading_score * 100, 1),
            "signals": signals,
            "metrics": self.metrics,
        }
        return result
