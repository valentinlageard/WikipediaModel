# coding=utf-8

#  Imports

import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import json
import re
import itertools

# Global values

esotericityNoise = False  #  Which esotericiy noise ?
disinformersProportion = False  # Which proportion of disinformers among trolls ?
multipleRuns = False  # Will the simulation have multiple runs ?
checkRe = "(\d+\.\d+|\d+)-(\d+\.\d+|\d+)-(\d+\.\d+|\d+)"  #  A regex expression used to check if the user wants to try various values for same parameters
trollRevert = False


# Class declarations

class User:  #  Create a class to create users, their properties and their actions
    def __init__(self, id, troll=False, disinformer=False, gaussianDistribution=False, mu=0, sigma=0,
                 activityDistribution=False):
        self.id = id  #  Attribute an id for this user
        #  Attribute a reliability to the user
        if not gaussianDistribution:
            self.reliability = random.random()
        else:
            self.reliability = np.random.normal(mu, sigma)
            while not 0 <= self.reliability <= 1:
                self.reliability = np.random.normal(mu, sigma)
        # Attribute an activity degree to the user
        if not activityDistribution:
            self.activity = random.random()
        else:
            self.activity = np.random.pareto(activityDistribution)
            while not self.activity <= 1:
                self.activity = np.random.pareto(activityDistribution)
        #  Define if the user is a troll or an honest user
        self.troll = troll
        #  Define if the user is a disinformer or a bullshitter
        self.disinformer = disinformer

    # Method of a user to check the truth value of an information
    # If the user's reliability added to a random number is superior to the informations esotericity, then the user knows its truth value
    def checkInfo(self, info):
        global esotericityNoise
        truth = info.truth if random.uniform(0, esotericityNoise) + self.reliability >= info.esotericity else None
        return truth

    # Method of a user to contribute
    #  Returns a list of informations
    # If the user is a troll, all the informations contributed will be false
    # If the user is an honest user, each information contributed will be true or false depending on her reliability
    # The esotericity of each information will be the sum of the user's reliability added to a random number between 0 and the esotericity noise
    def contribute(self, actionsNumber, actionsTurn):
        global infoCounter
        infosAdded = []
        if self.troll:
            if not self.disinformer:
                actionsNumber = random.randint(1, actionsTurn * 5)
            for i in range(actionsNumber):
                infoTruth = False
                infoEsotericity = random.uniform(0, esotericityNoise) + self.reliability
                info = Information(infoCounter, self.id, infoTruth, infoEsotericity)
                infosAdded.append(info)
                infoCounter += 1
        else:
            for i in range(actionsNumber):
                infoTruth = True if random.random() <= self.reliability else False
                infoEsotericity = random.uniform(0, esotericityNoise) + self.reliability
                info = Information(infoCounter, self.id, infoTruth, infoEsotericity)
                infosAdded.append(info)
                infoCounter += 1
        return infosAdded

    # Method of a user to check informations and delete them
    # Returns a list of information to be deleted
    #  A bullshitter can delete between 1 and all the informations in the article
    #  Honests users and disinformers will stop checking informations when they have selected actionsNumber informations to be deleted or when they have checked all the article
    #  If the user is honest, she will delete only information when she knows its truth value to be false
    #  If the user is a disinformer, she will delete only information when she knows its truth value to be true
    #  If the user is a bullshitter, she will delete all the randomly selected informations without checking their truth value
    def checkArticleAndDelete(self, currentArticle, actionsNumber):
        delete = []
        if actionsNumber >= len(currentArticle):
            actionsNumber = len(currentArticle)
        articleCopy = currentArticle[:]
        if self.troll and not self.disinformer:
            actionsNumber = random.randint(0, len(articleCopy))
        while actionsNumber > 0 and len(articleCopy) > 0:
            info = random.choice(articleCopy)
            if not self.troll:
                truthCheck = self.checkInfo(info)
                if not truthCheck and truthCheck is not None:
                    delete.append(info)
                    actionsNumber -= 1
            else:
                if self.disinformer:
                    truthCheck = self.checkInfo(info)
                    if truthCheck:
                        delete.append(info)
                        actionsNumber -= 1
                else:
                    delete.append(info)
                    actionsNumber -= 1
            articleCopy.remove(info)
        return delete

    def checkDiffMinus(self, infos, actionsTurn):
        report = False

        if len(infos) == 0:
            return 0, 0, 0, False

        if len(infos) > actionsTurn:
            report = True

        true = 0
        false = 0
        unknown = 0
        for info in infos:
            truthCheck = self.checkInfo(info)
            if truthCheck:
                true += 1
            elif not truthCheck and truthCheck is not None:
                false += 1
            else:
                unknown += 1

        if true > 0 and not report:
            report = True

        return true, false, unknown, report

    def checkDiffPlus(self, infos, actionsTurn):
        report = False

        if len(infos) == 0:
            return 0, 0, 0, False

        if len(infos) > actionsTurn:
            report = True

        true = 0
        false = 0
        unknown = 0
        for info in infos:
            truthCheck = self.checkInfo(info)
            if not truthCheck and truthCheck is not None:
                false += 1
            elif truthCheck:
                true += 1
            else:
                unknown += 1
        if false == len(infos) and not report:
            report = True

        return true, false, unknown, report

    def checkDiffs(self, diffHistory, actionsNumber, actionsTurn):
        toReport = []
        toRevert = None
        if len(diffHistory) <= actionsNumber:
            diffsToCheck = diffHistory[:0:-1]
        else:
            diffsToCheck = diffHistory[:len(diffHistory) - actionsNumber - 1:-1]
        if not self.troll:
            totalTruth = []
            totalFalse = []
            totalUnknown = []
            bullshitterDetected = [False for i in range(len(diffsToCheck))]

            for n, diff in enumerate(diffsToCheck):

                if diff.type == '-':
                    true, false, unknown, report = self.checkDiffMinus(diff.infos, actionsTurn)
                    totalTruth.append(true)
                    totalFalse.append(false)
                    totalUnknown.append(unknown)
                    if report:
                        toReport.append(diff)
                    if len(diff.infos) > actionsTurn:
                        bullshitterDetected[n] = True

                elif diff.type == '+':  #  If diff is a contribution
                    true, false, unknown, report = self.checkDiffPlus(diff.infos, actionsTurn)
                    totalTruth.append(-true)
                    totalFalse.append(-false)
                    totalUnknown.append(-unknown)
                    if report:
                        toReport.append(diff)
                    if len(diff.infos) > actionsTurn:
                        bullshitterDetected[n] = True

                elif diff.type == 'r':
                    infosDiffPlus = diff.infos[0]
                    infosDiffMinus = diff.infos[1]
                    truePlus, falsePlus, unknownPlus, reportPlus = self.checkDiffPlus(infosDiffPlus, actionsTurn)
                    trueMinus, falseMinus, unknownMinus, reportMinus = self.checkDiffMinus(infosDiffMinus,
                                                                                           actionsTurn)
                    totalTruth.append(trueMinus - truePlus)
                    totalFalse.append(falseMinus - falsePlus)
                    totalUnknown.append(unknownMinus - unknownPlus)
                    if diff.version - diff.reversedVersion > 25:
                        toReport.append(diff)
                        # break # Let or no ?

            totalUnknownStacked = []
            unknownSum = 0
            for i in range(len(totalUnknown)):
                if not bullshitterDetected[i]:
                    unknownSum -= abs(totalUnknown[i])
                else:
                    unknownSum += abs(totalUnknown[i])
                totalUnknownStacked.append(unknownSum)


            totalTruthStacked = [sum(totalTruth[:i + 1]) for i in range(len(totalTruth))]
            totalFalseStacked = [sum(totalFalse[:i + 1]) for i in range(len(totalFalse))]

            revertBenefits = [totalTruthStacked[i] - totalFalseStacked[i] + totalUnknownStacked[i] for i in
                              range(len(totalTruthStacked))]

            if len(revertBenefits) > 0:
                if max(revertBenefits) > 0:
                    i = revertBenefits.index(max(revertBenefits))
                    toRevert = diffHistory[len(diffHistory) - i - 1].version - 1
                else:
                    toRevert = None
            else:
                toRevert = None

        if self.troll:

            if not self.disinformer:
                if len(diffHistory) > 1:
                    toRevert = random.randint(0, len(diffHistory) - 1)

            else:
                totalTruth = []
                totalFalse = []
                totalUnknown = []

                for n, diff in enumerate(diffsToCheck):

                    if diff.type == '-':  #  If diff is a checkanddelete
                        true, false, unknown, report = self.checkDiffMinus(diff.infos, actionsTurn)
                        totalTruth.append(true)
                        totalFalse.append(false)
                        totalUnknown.append(unknown)

                    elif diff.type == '+':  #  If diff is a contribution
                        true, false, unknown, report = self.checkDiffPlus(diff.infos, actionsTurn)
                        totalTruth.append(-true)
                        totalFalse.append(-false)
                        totalUnknown.append(-unknown)

                    elif diff.type == 'r': #  If diff is a revert
                        infosDiffPlus = diff.infos[0]
                        infosDiffMinus = diff.infos[1]
                        truePlus, falsePlus, unknownPlus, reportPlus = self.checkDiffPlus(infosDiffPlus,
                                                                                          actionsTurn)
                        trueMinus, falseMinus, unknownMinus, reportMinus = self.checkDiffMinus(infosDiffMinus,
                                                                                               actionsTurn)
                        totalTruth.append(trueMinus - truePlus)
                        totalFalse.append(falseMinus - falsePlus)
                        totalUnknown.append(unknownMinus - unknownPlus)
                        # break # Let or no ?

                totalUnknownStacked = []
                unknownSum = 0
                for i in range(len(totalUnknown)):
                    unknownSum -= abs(totalUnknown[i])
                    totalUnknownStacked.append(unknownSum)

                totalTruthStacked = [sum(totalTruth[:i + 1]) for i in range(len(totalTruth))]
                totalFalseStacked = [sum(totalFalse[:i + 1]) for i in range(len(totalFalse))]

                revertBenefits = [totalFalseStacked[i] - totalTruthStacked[i] + totalUnknownStacked[i] for i in
                                  range(len(totalTruthStacked))]

                if len(revertBenefits) > 0:
                    if max(revertBenefits) > 0:
                        i = revertBenefits.index(max(revertBenefits))
                        toRevert = diffHistory[len(diffHistory) - i - 1].version - 1
                    else:
                        toRevert = None
                else:
                    toRevert = None

        if not toReport:
            toReport = None
        return toReport, toRevert

    def checkUser(self, diff, actionsTurn):
        if diff.type == '-':
            if len(diff.infos) > actionsTurn:
                return True
            for info in diff.infos:
                truthCheck = self.checkInfo(info)
                if truthCheck:
                    return True
        if diff.type == '+':
            if len(diff.infos) > actionsTurn:
                return True
            falseInfosCounter = 0
            for info in diff.infos:
                truthCheck = self.checkInfo(info)
                if truthCheck:
                    break
                elif not truthCheck and truthCheck is not None:
                    falseInfosCounter += 1
            if falseInfosCounter == len(diff.infos):
                return True
        if diff.type == 'r':
            if diff.version - diff.reversedVersion > actionsTurn:
                return True
        return False

    def __str__(self):
        return 'user n°' + str(self.id) + '[troll:' + str(self.troll) + ' ; disinformer:' + str(
            self.disinformer) + ' ; reliability:' + str(self.reliability) + ' ; activity:' + str(self.activity) + ']'


class Information:  #  Create a class to create information and their properties
    def __init__(self, infoId, userId, infoTruth, infoEsotericity):
        self.id = infoId
        self.author = userId
        self.truth = infoTruth
        self.esotericity = infoEsotericity

    def __str__(self):
        return "Info [Truth:" + str(self.truth) + "; infoId:" + str(self.id) + " ; author:" + str(
                self.author) + " ; esotericity:" + str(self.esotericity) + ']'


class Diff:  #   Create a class to create diffs
    def __init__(self, userId, type, version, infos=None, reversedVersion=None):
        self.userId = userId  #   The user that created this diff
        self.type = type  #   The type of the diff ('+' = contribution ; '-' = delete ; 'r' = revert)
        self.version = version
        self.infos = infos
        self.reversedVersion = reversedVersion

    def __str__(self):
        return 'Diff n°' + str(self.version) + ' [user:' + str(self.userId) + ' ; type:' + str(self.type) + ']'


# Functions

def fRange(start, stop, step):
    values = []
    i = 0
    while start + i * step < stop:
        values.append(float("{0:.2f}".format(start + i * step)))
        i += 1
    values.append(float(stop))
    return values


def askValue(question, floatValue=False, booleanValue=False):
    global checkRe, multipleRuns
    value = input(question)
    if value is '':
        return False
    elif not re.search(checkRe, value):
        if floatValue:
            value = float(value)
        elif booleanValue:
            value = bool(value)
        else:
            value = int(value)
        return value
    else:
        multipleRuns = True
        nvalues = []
        nvalue = []
        characters = list(value)
        for i in range(len(characters)):
            character = characters[i]
            if character == "-" or i == len(characters) - 1:
                if i == len(characters) - 1:
                    nvalue.append(character)
                nvalue = ''.join(nvalue)
                if floatValue:
                    nvalue = float(nvalue)
                elif booleanValue:
                    nvalue = bool(nvalue)
                else:
                    nvalue = int(nvalue)
                nvalues.append(nvalue)
                nvalue = []
            else:
                nvalue.append(character)
        if floatValue:
            values = fRange(nvalues[0], nvalues[1], nvalues[2])
        else:
            values = list(range(nvalues[0], nvalues[1] + 1, nvalues[2]))
        return values


def askParameters():
    global disinformersProportion, trollRevert
    gaussianDistribution = askValue('Gaussian distribution of reliability (uniform if empty) ? : ', booleanValue=True)
    mu = 0
    sigma = 0
    if gaussianDistribution:
        mu = askValue('Mu [0-1] (0.5) : ', floatValue=True)
        sigma = askValue('Sigma [0.01-1] (0.12) : ', floatValue=True)
    usersNumber = askValue('Number of sincere users : ')  #  How much users in the simulation ?
    trollNumber = askValue('Number of trolls : ')  #  How much trolls ?
    steps = askValue('Stop at the version : ')  #  How much simulation steps ?
    actionsTurn = askValue('Max number of actions at each turn : ')
    actionsDistribution = askValue('Pareto distribution of number of actions (uniform if empty) (0.5) : ',
                                   floatValue=True)
    activityDistribution = askValue('Pareto distribution of activity (uniform if empty) ? (5) : ', floatValue=True)
    esotericityNoise = askValue('Esotericity noise (1 if left empty) : ', floatValue=True)
    withRevert = askValue('With revert ? (leave empty if not) :', booleanValue=True)
    contributeProportion = askValue('Percentage of contribution for honest users (equiprobable if empty) : ',
                                    floatValue=True)
    checkAndDeleteProportion = askValue('Percentage of checkAndDelete for honest users (equiprobable if empty) : ',
                                        floatValue=True)
    contributeProportionTrolls = askValue('Percentage of contribution for trolls (equiprobable if empty) : ',
                                          floatValue=True)
    checkAndDeleteProportionTrolls = askValue('Percentage of checkAndDelete for trolls (equiprobable if empty) : ',
                                              floatValue=True)
    adminsNumber = askValue("Number of admins : ")  #  How much admins ?
    newUsers = askValue('Percentage of chances of new users each turn [0-1] : ', floatValue=True)
    newUsersTrollProportion = askValue('Percentage of trolls in new users [0-1] : ', floatValue=True)
    disinformersProportion = askValue('Proportion of disinformers [0-1] : ', floatValue=True)
    trollRevert = askValue('Can troll revert : ', booleanValue=True)
    repeatSimulation = askValue('Repeat simulation : ')
    saveSimulation = askValue('Save the simulation ? (leave empty if not) : ', booleanValue=True)
    if not esotericityNoise:
        esotericityNoise = 1
    parametersList = [gaussianDistribution, mu, sigma, usersNumber, trollNumber, steps, actionsTurn, esotericityNoise,
                      withRevert, contributeProportion, checkAndDeleteProportion, adminsNumber, newUsers,
                      newUsersTrollProportion, disinformersProportion, trollRevert, repeatSimulation, saveSimulation,
                      contributeProportionTrolls, checkAndDeleteProportionTrolls, actionsDistribution,
                      activityDistribution]
    return parametersList


def generateUsers(usersNumber, trolls=0, gaussianDistribution=False, mu=0, sigma=0, admins=False,
                  activityDistribution=False):  #  Function designed to generate a list of users at the beginning of a simulation
    global disinformersProportion
    users = []  #  Empty list of users
    counter = 0
    if not admins:
        for i in range(usersNumber):  #  For the number of users
            counter += 1
            user = User(counter, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma,
                        activityDistribution=activityDistribution)  #  Create an user with the id i
            users.append(user)
    if admins:
        for i in range(usersNumber):
            counter += 1
            user = User(counter, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma,
                        activityDistribution=activityDistribution)
            users.append(user)
    if trolls != 0:
        if disinformersProportion is False:
            disinformersNumber = int(trolls / 2)
        else:
            disinformersNumber = int(trolls * disinformersProportion)
        bullshittersNumber = trolls - disinformersNumber
        for i in range(disinformersNumber):
            counter += 1
            troll = User(counter, troll=True, disinformer=True, gaussianDistribution=gaussianDistribution, mu=mu,
                         sigma=sigma, activityDistribution=activityDistribution)
            users.append(troll)
        for i in range(bullshittersNumber):
            counter += 1
            troll = User(counter, troll=True, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma,
                         activityDistribution=activityDistribution)
            users.append(troll)
    return users  #  Return the list of users


def getUser(userId, users):
    for user in users:
        if user.id == userId:
            return user
    return False


def act(user, currentArticle, articleHistory, diffHistory, withRevert, actionsTurn, admins, contributeProportion,
        checkAndDeleteProportion, contributeProportionTrolls, checkAndDeleteProportionTrolls, actionsDistribution):

    randomNumber = random.random()

    if user.troll:
        if trollRevert:
            if randomNumber <= contributeProportionTrolls:
                action = 1
            elif contributeProportionTrolls < randomNumber <= contributeProportionTrolls + checkAndDeleteProportionTrolls:
                action = 2
            else:
                action = 3
        else:
            action = 1 if randomNumber <= contributeProportionTrolls else 2
    else:
        if withRevert or admins:
            if randomNumber <= contributeProportion:
                action = 1
            elif contributeProportion < randomNumber <= contributeProportion + checkAndDeleteProportion:
                action = 2
            else:
                action = 3
        else:
            action = 1 if randomNumber <= contributeProportion else 2
    if not actionsDistribution:
        actionsNumber = random.randint(1, actionsTurn)  #  Then, the user perform 1 to n actions
    else:
        actionsNumber = round(np.random.pareto(actionsDistribution))
        while not 1 <= actionsNumber <= actionsTurn:
            actionsNumber = round(np.random.pareto(actionsDistribution))
    mod = False
    toReport = None
    toRevert = None
    newDiff = None
    # What action the user perform ?
    if action == 1:  #  The user contributes
        contribution = user.contribute(actionsNumber, actionsTurn)  # Get the contribution of this user
        currentArticle.extend(contribution)  #  Add the contribution to the current version of the article
        newDiff = Diff(user.id, '+', len(articleHistory), contribution)  #  Add the diff to the history of diffs
        if contribution:
            mod = True
    elif action == 2:  #  The user checks a part of the article and deletes informations
        delete = user.checkArticleAndDelete(currentArticle,
                                            actionsNumber)  #  The list of infos to delete according to this user
        if delete:  #  If there are infos to delete
            diff = []
            for info in delete:  #  For each information to delete
                diff.append(info)
                currentArticle.remove(info)  #  Remove the info of the current version of the article
            newDiff = Diff(user.id, '-', len(articleHistory), diff)  #  Add the diff to the history of diffs
            mod = True
    elif action == 3:
        toReport, toRevert = user.checkDiffs(diffHistory, actionsNumber, actionsTurn)
        if withRevert:
            if toRevert:
                currentArticle = articleHistory[toRevert]
                infosDiffPlus = [info for info in currentArticle if info not in articleHistory[-1]]
                infosDiffMinus = [info for info in articleHistory[-1] if info not in currentArticle]
                newDiff = Diff(user.id, 'r', len(articleHistory), infos=[infosDiffPlus, infosDiffMinus],
                               reversedVersion=toRevert)
                mod = True
    return mod, currentArticle, newDiff, toReport, toRevert

# Simulation

def runWikiSimulation(steps, users, admins, actionsTurn, newUsers, newUsersTrollProportion, gaussianDistribution, mu,
                      sigma, withRevert, contributeProportion, checkAndDeleteProportion, contributeProportionTrolls,
                      checkAndDeleteProportionTrolls, actionsDistribution):  #  Function designed to run the simulation
    global disinformersProportion
    articleHistory = []  #  Create a list of versions of the article throughout its history
    diffHistory = []  #  Create a list of diffs of the article throughout its history
    userHistory = []
    reports = []
    bannedUsers = []
    usersNumber = len(users)
    trollsPercentage = newUsersTrollProportion
    while len(articleHistory) < steps:  #  For the number of steps
        if not multipleRuns:
            print(str(len(articleHistory)) + "/" + str(steps))
        random.shuffle(users)  #  Shuffle the order of users
        if len(users) == 0:
            if random.random() <= trollsPercentage:
                usersNumber += 1
                if random.random() <= disinformersProportion:
                    users.append(
                        User(usersNumber, troll=True, disinformer=True, gaussianDistribution=gaussianDistribution,
                             mu=mu, sigma=sigma))
                else:
                    users.append(
                        User(usersNumber, troll=True, disinformer=False, gaussianDistribution=gaussianDistribution,
                             mu=mu, sigma=sigma))
            else:
                usersNumber += 1
                users.append(User(usersNumber, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma))
        for user in users:  #  For each users
            # If the user is active (when a random number is below her activity threshold)
            if random.random() <= user.activity:
                # Get the current article
                try:
                    currentArticle = articleHistory[-1][:]
                except IndexError:
                    currentArticle = []
                mod, newVersion, newDiff, toReport, toRevert = act(user, currentArticle, articleHistory, diffHistory,
                                                                   withRevert, actionsTurn, admins,
                                                                   contributeProportion, checkAndDeleteProportion,
                                                                   contributeProportionTrolls,
                                                                   checkAndDeleteProportionTrolls, actionsDistribution)
                if mod:
                    articleHistory.append(newVersion)
                    diffHistory.append(newDiff)
                    userHistory.append(users[:])
                if toReport is not None:
                    reports.extend(toReport)
            if reports and admins:
                for admin in admins:
                    for report in reports:
                        userId = report.userId
                        user = getUser(userId, users)
                        if not user:
                            reports.remove(report)
                        else:
                            trueReport = admin.checkUser(report, actionsTurn)
                            if trueReport:
                                users.remove(user)
                                bannedUsers.append(user)
                                reports.remove(report)
                            else:
                                reports.remove(report)
            if len(articleHistory):
                break
        if newUsers is not False and random.random() <= newUsers:
            if random.random() <= trollsPercentage:
                usersNumber += 1
                if random.random() <= disinformersProportion:
                    users.append(
                        User(usersNumber, troll=True, disinformer=True, gaussianDistribution=gaussianDistribution,
                             mu=mu, sigma=sigma))
                else:
                    users.append(
                        User(usersNumber, troll=True, disinformer=False, gaussianDistribution=gaussianDistribution,
                             mu=mu, sigma=sigma))
            else:
                usersNumber += 1
                users.append(User(usersNumber, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma))
    return articleHistory, diffHistory, bannedUsers, userHistory  #  Return the history of versions and diffs of the article

# Prints and filename


def printGraph(articleHistory, userHistory):
    # Get information about informations
    trueInfos = []
    falseInfos = []
    for version in articleHistory:
        trueInfosVersion = 0
        falseInfosVersion = 0
        for info in version:
            if info.truth:
                trueInfosVersion += 1
            else:
                falseInfosVersion += 1
        trueInfos.append(trueInfosVersion)
        falseInfos.append(falseInfosVersion)
    #  Get information about the history of users
    trollsHistory = []
    sincereUsersHistory = []
    for users in userHistory:
        trollsNumber = 0
        sincereUsersNumber = 0
        for user in users:
            if user.troll:
                trollsNumber += 1
            else:
                sincereUsersNumber += 1
        trollsHistory.append(trollsNumber)
        sincereUsersHistory.append(sincereUsersNumber)
    # Create curves
    x = np.arange(len(articleHistory))
    y1 = np.array(trueInfos)
    y2 = y1 + np.array(falseInfos)
    y2b = np.array(falseInfos)
    y4 = np.array(trollsHistory)
    y5 = np.array(sincereUsersHistory)
    # plt.fill_between(x,y1,0,color='blue')
    # plt.fill_between(x,y1,y2,color='red')
    plt.plot(x, y1, color='blue')
    plt.plot(x, y2b, color='red')
    plt.plot(x, y4, color='darkred')
    plt.plot(x, y5, color='darkblue')
    plt.show()


def printDiff(diffHistory):
    checking = True
    while checking:
        diffToCheckValue = input('Which diff do you want to check ? : ')
        if diffToCheckValue == '':
            checking = False
        else:
            diffToCheckValue = int(diffToCheckValue)
            diffToCheck = diffHistory[diffToCheckValue]
            print(diffToCheck)
            if diffToCheck.type != "r":
                for info in diffToCheck.infos:
                    print(info)
            else:
                print("Reversed version : " + str(diffToCheck.reversedVersion))
                print("Diffs reversed : ")
                diffsToPrint = diffHistory[diffToCheck.reversedVersion + 1:diffToCheck.version]
                for diff in diffsToPrint:
                    print("    " + str(diff))
                    for info in diff.infos:
                        print("    " + str(info))


def getFileName(parameters):
    parametersInitials = ["G", "Mu", "Sig", "HU", "TU", "S", "AT", "EN", "RV", "C", "C&D", "AU", "NU", "NUT",
                          "Dis", "TR",
                          "V", "S", "CT", "C&DT", "AD", "ACD"]
    fileName = ['runs/']
    cleanedParameters = []
    for parameter in parameters:
        if len(parameter) > 1:
            cleanedParameters.append(str(parameter[0]) + '-' + str(parameter[-1]))
        else:
            cleanedParameters.append(str(parameter[0]))
    for i in range(len(parameters)):
        if cleanedParameters[i] is not 'False':
            if cleanedParameters[i] != '0' and not (i == 7 and cleanedParameters[i] == '1') and not i == 17:
                fileName.append(parametersInitials[i] + cleanedParameters[i])
    fileName.append('.json')
    return "".join(fileName)

# Run general

def runProgram(parameters):
    global multipleRuns, esotericityNoise, disinformersProportion, trollRevert
    gaussianDistribution = parameters[0]
    mu = parameters[1]
    sigma = parameters[2]
    usersNumber = parameters[3]
    trollNumber = parameters[4]
    steps = parameters[5]
    actionsTurn = parameters[6]
    esotericityNoise = parameters[7]
    withRevert = parameters[8]
    contributeProportion = parameters[9]
    checkAndDeleteProportion = parameters[10]
    adminsNumber = parameters[11]
    newUsers = parameters[12]
    newUsersTrollProportion = parameters[13]
    disinformersProportion = parameters[14]
    trollRevert = parameters[15]
    repeatSimulation = parameters[16]
    saveSimulation = parameters[17]
    contributeProportionTrolls = parameters[18]
    checkAndDeleteProportionTrolls = parameters[19]
    actionsDistribution = parameters[20]
    activityDistribution = parameters[21]

    if disinformersProportion is False:
        disinformersProportion = 0.5

    if not withRevert and not adminsNumber:
        if not contributeProportion:
            if not checkAndDeleteProportion:
                contributeProportion = 0.5
                checkAndDeleteProportion = 0.5
            else:
                contributeProportion = 1 - checkAndDeleteProportion
        else:
            if not checkAndDeleteProportion:
                checkAndDeleteProportion = 1 - contributeProportion
            else:
                contributeProportion = 0.5
                checkAndDeleteProportion = 0.5
    else:
        if not contributeProportion:
            if not checkAndDeleteProportion:
                contributeProportion = 1 / 3
                checkAndDeleteProportion = 1 / 3
            else:
                contributeProportion = (1 - checkAndDeleteProportion) / 2
        else:
            if not checkAndDeleteProportion:
                checkAndDeleteProportion = (1 - contributeProportion) / 2

    if not trollRevert:
        if not contributeProportionTrolls:
            if not checkAndDeleteProportionTrolls:
                contributeProportionTrolls = 0.5
                checkAndDeleteProportionTrolls = 0.5
            else:
                contributeProportionTrolls = 1 - checkAndDeleteProportionTrolls
        else:
            if not checkAndDeleteProportionTrolls:
                checkAndDeleteProportionTrolls = 1 - contributeProportionTrolls
            else:
                contributeProportionTrolls = 0.5
                checkAndDeleteProportionTrolls = 0.5
    else:
        if not contributeProportionTrolls:
            if not checkAndDeleteProportionTrolls:
                contributeProportionTrolls = 1 / 3
                checkAndDeleteProportionTrolls = 1 / 3
            else:
                contributeProportionTrolls = (1 - checkAndDeleteProportionTrolls) / 2
        else:
            if not checkAndDeleteProportionTrolls:
                checkAndDeleteProportionTrolls = (1 - contributeProportionTrolls) / 2

    print('\nGenerating users...')
    users = generateUsers(usersNumber, trollNumber, gaussianDistribution=gaussianDistribution, mu=mu, sigma=sigma,
                          activityDistribution=activityDistribution)
    admins = False
    if adminsNumber != 0:  #  If there are admins
        admins = generateUsers(adminsNumber, admins=True, gaussianDistribution=gaussianDistribution, mu=mu,
                               sigma=sigma, activityDistribution=activityDistribution)  #  Generate admins
    # Run the simulation and get infos
    print('Running simulation n°' + str(repeatSimulation) + '...')
    articleHistory, diffHistory, bannedUsers, userHistory = runWikiSimulation(steps, users, admins,
                                                                                                actionsTurn, newUsers,
                                                                                                newUsersTrollProportion,
                                                                                                gaussianDistribution,
                                                                                                mu, sigma, withRevert,
                                                                                                contributeProportion,
                                                                                                checkAndDeleteProportion,
                                                                                                contributeProportionTrolls,
                                                                                                checkAndDeleteProportionTrolls,
                                                                                                actionsDistribution)

    # Print stuff

    if saveSimulation:
        print('Constructing data...')
        run = {'gaussianDistribution': gaussianDistribution, 'mu': mu, 'sigma': sigma, 'usersNumber': usersNumber,
               'trollNumber': trollNumber, 'steps': steps, 'actionsTurn': actionsTurn, 'withRevert': withRevert,
               'adminsNumber': adminsNumber, 'newUsers': newUsers,
               'newUsersTrollProportion': newUsersTrollProportion, 'version': repeatSimulation,
               'esotericityNoise': esotericityNoise, 'contributeProportion': contributeProportion,
               'checkAndDeleteProportion': checkAndDeleteProportion, 'disinformersProportion': disinformersProportion,
               'trollRevert': trollRevert, 'contributeProportionTrolls': contributeProportionTrolls,
               'checkAndDeleteProportionTrolls': checkAndDeleteProportionTrolls,
               'actionsDistribution': actionsDistribution, 'activityDistribution': activityDistribution}
        allInfos = []
        diffHistoryConverted = []

        for i, diff in enumerate(diffHistory):
            diffConverted = {'type': diff.type, 'author': diff.userId, 'version': diff.version,
                             'reversedVersion': diff.reversedVersion}
            if diff.type == '+':
                addedInfosId = []
                for info in diff.infos:
                    allInfos.append(
                        {'id': info.id, 'truth': info.truth, 'author': info.author, 'esotericity': info.esotericity})
                    addedInfosId.append(info.id)
                diffConverted['addedInfos'] = addedInfosId
            elif diff.type == '-':
                deletedInfosId = []
                for info in diff.infos:
                    deletedInfosId.append(info.id)
                diffConverted['deletedInfos'] = deletedInfosId
            else:
                addedInfosId = []
                deletedInfosId = []
                for info in diff.infos[0]:
                    addedInfosId.append(info.id)
                for info in diff.infos[1]:
                    deletedInfosId.append(info.id)
                diffConverted['addedInfos'] = addedInfosId
                diffConverted['deletedInfos'] = deletedInfosId
            diffHistoryConverted.append(diffConverted)

        run['All Infos'] = allInfos
        run['Diff History'] = diffHistoryConverted

        allUsers = []
        alreadyStored = []
        userHistoryConverted = []
        for i, users in enumerate(userHistory):
            usersList = []
            for user in users:
                if allUsers:
                    if not user.id in alreadyStored:
                        allUsers.append({'id': user.id, 'reliability': user.reliability, 'activity': user.activity,
                                         'troll': user.troll, 'disinformer': user.disinformer})
                        alreadyStored.append(user.id)
                else:
                    allUsers.append(
                        {'id': user.id, 'reliability': user.reliability, 'activity': user.activity, 'troll': user.troll,
                         'disinformer': user.disinformer})
                    alreadyStored.append(user.id)
                usersList.append(user.id)
            userHistoryConverted.append(usersList)

        run['All Users'] = allUsers
        run['Users History'] = userHistoryConverted

        return run

    print('Creating graph...')

    p = Process(target=printGraph, args=(articleHistory, userHistory))
    p.start()

    printDiff(diffHistory)

# Run the software

parameters = askParameters()

if multipleRuns:
    for i in range(len(parameters)):
        if not isinstance(parameters[i], list):
            parameters[i] = [parameters[i]]
    parametersMatrix = []
    for parametersList in itertools.product(*parameters):
        parametersCombination = []
        for parameter in parametersList:
            parametersCombination.append(parameter)
        parametersMatrix.append(parametersCombination)

    data = []
    for i in range(len(parametersMatrix)):
        specialParameters = parametersMatrix[i]
        run = runProgram(specialParameters)
        data.append(run)
else:
    data = runProgram(parameters)
    parameters = [[parameter] for parameter in parameters]

if parameters[18]:
    print('\nSaving data...')
    fileName = getFileName(parameters)
    f = open("".join(fileName), 'w')
    json.dump(data, f)
